import sys
import genotypes
from graphviz import Digraph


def plot(genotype, filename, graph_name):
    g = Digraph(
        format='png',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("x_{t}", fillcolor='darkseagreen2')
    g.node("h_{t-1}", fillcolor='darkseagreen2')
    g.node("0", fillcolor='lightblue')
    g.edge("x_{t}", "0", fillcolor="gray")
    g.edge("h_{t-1}", "0", fillcolor="gray")
    steps = len(genotype)

    for i in range(1, steps + 1):
        g.node(str(i), fillcolor='lightblue')

    for i, (op, j) in enumerate(genotype):
        g.edge(str(j), str(i + 1), label=op, fillcolor="gray")

    g.node("h_{t}", fillcolor='palegoldenrod')
    for i in range(1, steps + 1):
        g.edge(str(i), "h_{t}", fillcolor="gray")

    g.body.append(r'label = ' + graph_name)
    g.body.append('fontsize=20')

    g.render(filename, view=False)


def plot_lstm(filename):
    g = Digraph(
        format='png',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2',
                       fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("x_{t}", fillcolor='darkseagreen2')
    g.node("h_{t-1}", fillcolor='darkseagreen2')
    g.node("c_{t-1}", fillcolor='darkseagreen2')
    g.node("0", fillcolor='lightblue')
    g.node("f", fillcolor='lightblue')
    g.node("i", fillcolor='lightblue')
    g.node("c", fillcolor='lightblue')
    g.node("o", fillcolor='lightblue')
    g.node("h_{t}", fillcolor='palegoldenrod')
    g.node("c_{t}", fillcolor='palegoldenrod')

    g.edge("x_{t}", "0", fillcolor="gray")
    g.edge("h_{t-1}", "0", fillcolor="gray")

    g.edge("0", "f", fillcolor="gray", label='sigmoid')
    g.edge("0", "i", fillcolor="gray", label='sigmoid')
    g.edge("0", "c", fillcolor="gray", label='tanh')
    g.edge("0", "o", fillcolor="gray", label='sigmoid')

    g.edge("c_{t-1}", "c_{t}", fillcolor="gray")
    g.edge("f", "c_{t}", fillcolor="gray")
    g.edge("i", "c_{t}", fillcolor="gray")
    g.edge("c", "c_{t}", fillcolor="gray")

    g.edge("c_{t}", "h_{t}", label='tanh', fillcolor="gray")
    g.edge("o", "h_{t}", fillcolor="gray")
    g.render(filename, view=False)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
        sys.exit(1)

    genotype_name = sys.argv[1]
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)

    # plot(genotype.recurrent, "foo", 'epoch_1')
    plot_lstm('cell_lstm')
