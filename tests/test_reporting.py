from xml.etree import ElementTree

from hybrid_training_analytics.reporting import _run_chart


def test_graph_has_break_for_temporal_gap(run):
    run.samples = [s for s in run.samples if not 500 < s.elapsed_s < 700]
    svg = _run_chart(run, False)
    root = ElementTree.fromstring(svg)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    plotted = [
        path
        for path in root.findall(".//svg:path", namespace)
        if "#0f766e" in path.attrib.get("style", "")
    ]
    assert len(plotted) == 1
    assert plotted[0].attrib["d"].count("M ") >= 2
