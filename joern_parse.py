import argparse
import logging
import os
import re
import socket
import subprocess
import time

from cpgqls_client import CPGQLSClient, import_code_query
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LIST_PATTERN = re.compile(r'List\((.*?)\)', re.DOTALL)
TRIPLE_QUOTED_PATTERN = re.compile(r'"""\s*(.*?)\s*"""', re.DOTALL)


def start_joern_server(joern_path, server_endpoint, timeout):
    try:
        subprocess.Popen(
            [joern_path + 'joern', '--server'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True
        )
        logger.info("Waiting for Joern server to start...")

        host, port = server_endpoint.split(':')
        port = int(port)
        time.sleep(10)
        start_time = time.time()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            while s.connect_ex((host, port)) != 0:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Joern server did not start within the timeout period.")
                time.sleep(1)
        logger.info("Joern server started successfully.")
    except Exception as e:
        logger.error(f"Error starting Joern server: {e}")
        raise


def stop_joern_server():
    try:
        subprocess.run(['pkill', '-f', 'joern.*--server'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Joern server stopped successfully.")
    except Exception as e:
        logger.error(f"Error stopping Joern server: {e}")
        raise


def parse_and_export(code_dir, client):
    os.makedirs(f'{code_dir}/output', exist_ok=True)

    try:
        # query = import_code_query(code_dir)
        # client.execute(query)
        query = f'importCpg(\"workspace/{code_dir}/cpg.bin.zip\")'
        client.execute(query)

        query = 'cpg.method.isExternal(false).name.l'
        result = client.execute(query)
        methods_str = LIST_PATTERN.search(result['stdout']).group(1)
        user_defined_methods = [method.strip('"') for method in methods_str.split(', ') if method != '"<global>"']
        if not user_defined_methods:
            logger.warning(f"No user-defined methods found in {code_dir}. Skipping PDG extraction.")
            return

        query = 'cpg.all.toJsonPretty'
        result = client.execute(query)
        nodes_str = TRIPLE_QUOTED_PATTERN.search(result['stdout']).group(1)
        with open(f'{code_dir}/output/nodes.json', 'w', encoding='utf-8') as file:
            file.write(nodes_str)

        graphs = ['pdg', 'ddg', 'cdg', 'cfg', 'ast']
        for method in user_defined_methods:
            for graph in graphs:
                query = f'cpg.method("{method}").dot{graph.capitalize()}.l'
                result = client.execute(query)
                dot_content = TRIPLE_QUOTED_PATTERN.search(result['stdout']).group(1)
                with open(f'{code_dir}/output/{method}_{graph}.dot', 'w', encoding='utf-8') as file:
                    file.write(dot_content)

        client.execute('delete')
        client.execute('close')

    except Exception as e:
        logger.error(f"An error occurred while process code directory {code_dir}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joern_path', type=str, default='~/bin/joern/joern-cli/',
                        help="Path to the Joern executable.")
    parser.add_argument('--server_endpoint', type=str, default="localhost:8080",
                        help="Server endpoint in the format host:port.")
    parser.add_argument('--timeout', type=int, default=30,
                        help="Timeout in seconds for waiting for Joern server to start.")
    parser.add_argument('--data_dir', type=str, default='~/vrepair_bug_data/train/',
                        help="The input data directory.")
    parser.add_argument('--batch_size', type=int, default=100,
                        help="How many projects to process before restarting Joern.")
    args = parser.parse_args()

    joern_path = os.path.expanduser(args.joern_path)
    server_endpoint = args.server_endpoint
    timeout = args.timeout
    data_dir = os.path.expanduser(args.data_dir)
    batch_size = args.batch_size

    os.chdir(data_dir)
    code_dirs = sorted([_dir for _dir in os.listdir() if _dir.isdigit() and os.path.isdir(_dir)], key=int)

    logger.info("***** Generating code CPG *****")
    for code_dir in tqdm(code_dirs, total=len(code_dirs), position=0):
        cpg_path = f'workspace/{code_dir}/cpg.bin.zip'
        if not os.path.exists(cpg_path):
            os.makedirs(f'workspace/{code_dir}', exist_ok=True)
            subprocess.run([os.path.join(joern_path, 'c2cpg.sh'), '-J-Xmx16384m', code_dir, '--output',
                            cpg_path], check=True)

    start_joern_server(joern_path, server_endpoint, timeout)
    client = CPGQLSClient(server_endpoint)

    logger.info("***** Running code analysis *****")
    processed_count = 0
    for code_dir in tqdm(code_dirs, total=len(code_dirs), position=0):
        parse_and_export(code_dir, client)
        processed_count += 1
        if processed_count % batch_size == 0:
            logger.info(f"Restart Joern server to avoid OOM problem.")
            stop_joern_server()
            start_joern_server(joern_path, server_endpoint, timeout)
            client = CPGQLSClient(server_endpoint)
    logger.info("All projects processed successfully.")
    stop_joern_server()


if __name__ == "__main__":
    main()
