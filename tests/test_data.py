import tempfile
import unittest
from pathlib import Path

from python_pets.dataset import find_pairs, parse_voc
from python_pets.download import has_files
from python_pets.inference import summarize

VOC = """<annotation>
  <object><name>Cat</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>
  <object><name>horse</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>
  <object><name>dog</name><bndbox><xmin>5</xmin><ymin>5</ymin><xmax>5</xmax><ymax>9</ymax></bndbox></object>
</annotation>"""


class DataTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_parse_voc_keeps_known_labels_and_valid_boxes(self):
        xml = self.root / "a.xml"
        xml.write_text(VOC)
        self.assertEqual(parse_voc(xml), ([[1.0, 2.0, 30.0, 40.0]], [1]))

    def test_find_pairs_matches_by_name(self):
        raw, annotations = self.root / "raw", self.root / "annotations"
        raw.mkdir()
        annotations.mkdir()
        (raw / "a.png").touch()
        (raw / "unlabelled.png").touch()
        (annotations / "a.xml").touch()
        self.assertEqual([img.name for img, _ in find_pairs(raw, annotations)], ["a.png"])

    def test_has_files_ignores_gitkeep(self):
        (self.root / ".gitkeep").touch()
        self.assertFalse(has_files(self.root))
        (self.root / "a.png").touch()
        self.assertTrue(has_files(self.root))

    def test_summarize(self):
        self.assertEqual(summarize([])[0], "none")
        both = [{"label": "dog"}, {"label": "cat"}]
        self.assertEqual(summarize(both), ("both", {"cat": 1, "dog": 1}))


if __name__ == "__main__":
    unittest.main()
