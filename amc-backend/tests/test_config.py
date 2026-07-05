import os
import unittest
from amc.config import AppConfig
from amc.constants import ModulationType

class TestConfigAndImports(unittest.TestCase):
    
    def test_default_config_loading(self):
        # Locate the default config path relative to the workspace
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default_config.yaml")
        config_path = os.path.abspath(config_path)
        
        self.assertTrue(os.path.exists(config_path), f"Config file not found at: {config_path}")
        
        config = AppConfig.load_from_yaml(config_path)
        
        # Verify attributes match default YAML structure
        self.assertEqual(config.project.name, "Automatic Modulation Classification & Spectrum Analyzer")
        self.assertEqual(config.generator.samples_per_symbol, 8)
        self.assertEqual(config.analyzer.window, "hann")
        
    def test_constants(self):
        self.assertIn("BPSK", [m.value for m in ModulationType])
        self.assertIn("16QAM", [m.value for m in ModulationType])

if __name__ == "__main__":
    unittest.main()
