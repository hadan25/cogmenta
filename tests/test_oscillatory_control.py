import unittest
import time
from controller.oscillatory_control import OscillatoryController

class TestOscillatoryControl(unittest.TestCase):
    def setUp(self):
        self.controller = OscillatoryController()

    def test_oscillation_update(self):
        """Test oscillation updates"""
        amplitude = self.controller.update(dt=0.025)  # 1/40th of a second
        self.assertGreaterEqual(amplitude, 0)
        self.assertLessEqual(amplitude, 1)

    def test_frequency_modulation(self):
        """Test frequency modulation with attention"""
        initial_freq = self.controller.oscillations['gamma']['freq']
        self.controller.modulate_frequencies(attention_load=0.5)
        
        self.assertGreater(
            self.controller.oscillations['gamma']['freq'],
            initial_freq
        )

    def test_state_history(self):
        """Test state history recording"""
        self.controller.update(dt=0.025)
        self.assertEqual(len(self.controller.state_history), 1)
        self.assertIn('amplitude', self.controller.state_history[0])
        self.assertIn('phases', self.controller.state_history[0])
