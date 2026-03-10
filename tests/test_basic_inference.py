from tools.retriever import retrieve


def test_retrieve_stub_shape():
    result = retrieve("sensor tags")
    assert result["query"] == "sensor tags"
    assert isinstance(result["results"], list)
    assert result["status"] == "stub"
