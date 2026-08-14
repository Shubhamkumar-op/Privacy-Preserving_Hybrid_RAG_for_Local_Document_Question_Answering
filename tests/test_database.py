from src.storage.database import ChunkDatabase


def test_database_round_trip(tmp_path):
    db = ChunkDatabase(tmp_path / "chunks.db")
    db.add("paper.pdf", 3, "retrieval augmented generation")
    rows = db.search("retrieval")
    assert rows == [("paper.pdf", 3, "retrieval augmented generation")]
