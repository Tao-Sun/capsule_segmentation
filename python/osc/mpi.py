from mpi4py import MPI
import socket
import os
import argparse
import subprocess

FLAGS=None

"""
Start the ps or worker host in the tensorflow cluster.
"""

def _get_hosts(nodes, num_ps_hosts, hosts_domain, ps_port, worker_port):
    nodes = list(set(nodes.split(',')))

    ps_hosts = [nodes[i] + "." + hosts_domain + ":" + ps_port for i in range(num_ps_hosts)]
    worker_hosts = [nodes[i] + "." + hosts_domain + ":" + worker_port for i in range(num_ps_hosts, len(nodes))]

    return ps_hosts, worker_hosts


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ps_hosts, worker_hosts = _get_hosts(FLAGS.nodes, 
                                        FLAGS.num_ps_hosts, 
                                        FLAGS.hosts_domain, 
                                        FLAGS.ps_port,
                                        FLAGS.worker_port)
    print("I am rank " + str(rank) + "...")
    hostname = socket.gethostname()
    print("My hostname is: " + hostname)
    
    if (hostname+":"+FLAGS.ps_port) in ps_hosts:
        task_index = ps_hosts.index(hostname+":"+FLAGS.ps_port)
        print("My task index: ps" + str(task_index))
        job_name = "ps"
        print("init ps task...\n")
        ps_cmd = "python -u " + FLAGS.script + \
                 " --ps_hosts=" + ','.join(ps_hosts) + \
                 " --worker_hosts=" + ','.join(worker_hosts) + \
                 " --job_name=" + job_name + \
                 " --task_index=" + str(task_index) + \
                 " " + FLAGS.script_params
        subprocess.Popen(ps_cmd, shell=True)

    if (hostname+":"+FLAGS.worker_port) in worker_hosts:
        task_index = worker_hosts.index(hostname+":"+FLAGS.worker_port)
        print("My task index: worker" + str(task_index))
        job_name = "worker"
        print("init worker task... ")
        worker_cmd = "python -u " + FLAGS.script + \
                 " --ps_hosts=" + ','.join(ps_hosts) + \
                 " --worker_hosts=" + ','.join(worker_hosts) + \
                 " --job_name=" + job_name + \
                 " --task_index=" + str(task_index) + \
                 " " + FLAGS.script_params + \
                 " > " + FLAGS.log_file + "-" + str(task_index)
        print("worker init with cmd:")
        print(worker_cmd + "\n")
        subprocess.Popen(worker_cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nodes",
        type=str,
        default=""
    )
    parser.add_argument(
        "--hosts_domain",
        type=str,
        default="ten.osc.edu"
    )
    parser.add_argument(
        "--ps_port",
        type=str,
        default="2222"
    )
    parser.add_argument(
        "--worker_port",
        type=str,
        default="2223"
    )
    parser.add_argument(
        "--num_ps_hosts",
        type=int,
        default=1
    )
    parser.add_argument(
        "--script",
        type=str,
        default="",
        help="The .py file to execute in hosts"
    )
    parser.add_argument(
        "--script_params",
        type=str,
        default="",
        help="Params for the script."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help="Log file."
    )
    
    FLAGS, unparsed = parser.parse_known_args()
    main()