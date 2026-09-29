"""Offline regression checks for the portable generated robot asset layout."""

import json
import os
import shutil
import tempfile
import unittest

from a0509pp.models import load_manifest
from a0509pp.sim.asset_builder import ROBOTIQ_CFG_REL, bundle_robotiq, check_robotiq_lfs


class AssetBundleTests(unittest.TestCase):
    def test_bundle_survives_source_removal_and_relocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source")
            config = os.path.join(source, ROBOTIQ_CFG_REL)
            part = os.path.join(source, "grippers", "Robotiq_2F_85", "parts", "mesh.usdc")
            os.makedirs(os.path.dirname(config))
            os.makedirs(os.path.dirname(part))
            with open(config, "w") as f:
                f.write("#usda 1.0\n(def Xform \"Gripper\" (references = @../parts/mesh.usdc@) {})\n")
            with open(part, "wb") as f:
                f.write(b"fixture geometry")
            os.makedirs(os.path.join(source, ".git"))
            generated = os.path.join(tmp, "generated")
            bundled = bundle_robotiq(source, generated)
            self.assertFalse(os.path.exists(os.path.join(bundled, ".git")))
            with open(os.path.join(generated, "asset_manifest.json"), "w") as f:
                json.dump({"robotiq_usd": os.path.relpath(os.path.join(bundled, ROBOTIQ_CFG_REL), generated)}, f)
            shutil.rmtree(source)
            moved = os.path.join(tmp, "moved")
            shutil.move(generated, moved)
            usd = load_manifest(os.path.join(moved, "asset_manifest.json"))["robotiq_usd"]
            self.assertTrue(os.path.isfile(usd))
            self.assertTrue(os.path.isfile(os.path.join(os.path.dirname(usd), "../parts/mesh.usdc")))

    def test_lfs_pointer_is_rejected(self):
        with tempfile.TemporaryDirectory() as source:
            config = os.path.join(source, ROBOTIQ_CFG_REL)
            os.makedirs(os.path.dirname(config))
            with open(config, "wb") as f:
                f.write(b"version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(RuntimeError, "LFS pointer"):
                check_robotiq_lfs(source)


if __name__ == "__main__":
    unittest.main()
