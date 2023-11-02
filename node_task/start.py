#!/usr/bin/python
# encoding=utf-8
import datetime
import os
import threading


def execCmd(cmd):
    try:
        os.system(cmd)
    except:
        print('%s\t Runtime Error' % (cmd))


if __name__ == '__main__':
    if_parallel = False

    cmds = [
        "python main_node.py --lr 3e-3 --model gat --ss --num_layer 1 --lam 0.8",
        "python main_node.py --lr 3e-3 --model gat",
        "python main_node.py --lr 3e-3 --model gat --ca ",
        "python main_node.py --lr 2e-2 --model gcn",
        "python main_node.py --lr 2e-2 --model gcn --ca",
        "python main_node.py --lr 2e-2 --model gcn --ss --lam 8",
        "python main_node.py --lr 3e-3 --model sage",
        "python main_node.py --lr 3e-3 --model sage --ca",
        "python main_node.py --lr 3e-3 --model sage --ss --lam 0.8",
        "python main_node.py --lr 3e-3 --model cheb",
        "python main_node.py --lr 3e-3 --model cheb --ca",
        "python main_node.py --lr 3e-3 --model cheb --ss --lam 0.8",

        "python main_node.py --lr 3e-3 --model gat --task OGR",
        "python main_node.py --lr 3e-3 --model gat --ca --task OGR",
        "python main_node.py --lr 3e-4 --model gat --ss --task OGR --lam 0.8",
        "python main_node.py --lr 2e-2 --model gcn --task OGR",
        "python main_node.py --lr 2e-2 --model gcn --ca --task OGR",
        "python main_node.py --lr 2e-2 --model gcn --ss --task OGR --lam 0.8",
        "python main_node.py --lr 3e-3 --model sage --task OGR",
        "python main_node.py --lr 3e-3 --model sage --ca --task OGR",
        "python main_node.py --lr 3e-3 --model sage --ss --task OGR --lam 0.8",
        "python main_node.py --lr 3e-3 --model cheb --task OGR",
        "python main_node.py --lr 3e-3 --model cheb --ca --task OGR",
        "python main_node.py --lr 3e-3 --model cheb --ss --task OGR --lam 0.8",
    ]

    if if_parallel:
        threads = []
        for cmd in cmds:
            th = threading.Thread(target=execCmd, args=(cmd,))
            th.start()
            threads.append(th)

        for th in threads:
            th.join()
    else:
        for cmd in cmds:
            try:
                os.system(cmd)
            except:
                print('%s\t Runtime Error' % (cmd))
                exit()


