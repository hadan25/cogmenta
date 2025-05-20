import unittest
import numpy as np
import time
from cognitive.consciousness_field import ConsciousnessField

class TestConsciousnessField(unittest.TestCase):
    def setUp(self):
        self.field = ConsciousnessField(field_size=100)

    def test_field_initialization(self):
        """Test field initialization"""
        self.assertEqual(self.field.field_size, 100)
        self.assertEqual(len(self.field.field), 100)
        self.assertTrue(np.all(self.field.field == 0))

    def test_broadcast_and_decay(self):
        """Test broadcasting and decay"""
        # Broadcast content
        self.field.broadcast("test", 50, 1.0)
        max_val = np.max(self.field.field)
        self.assertGreater(max_val, 0.9)

        # Test decay
        self.field.update(dt=0.1)
        new_max = np.max(self.field.field)
        self.assertLess(new_max, max_val)

    def test_conscious_content(self):
        """Test conscious content extraction"""
        # Empty field
        content = self.field.get_conscious_content()
        self.assertEqual(content['coverage'], 0)

        # With content
        self.field.broadcast("test", 50, 1.0)
        content = self.field.get_conscious_content()
        self.assertGreater(content['coverage'], 0)
        self.assertGreater(content['mean_activation'], 0)
