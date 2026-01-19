import unittest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.agents import ChefProjetAgent, DataEngineerAgent
from backend.models.message import Message, MessageType
from backend.communication.message_bus import MessageBus


class TestMessageBus(unittest.TestCase):
    """Tests du bus"""
    
    def setUp(self):
        self.bus = MessageBus()
    
    def test_register(self):
        """Test d'enregistrement"""
        self.bus.register_agent("Test")
        self.assertIn("Test", self.bus.agent_queues)
    
    def test_send_receive(self):
        """Test d'envoi/réception"""
        self.bus.register_agent("A")
        self.bus.register_agent("B")
        
        msg = Message("A", "B", MessageType.TASK_REQUEST, {"test": "data"})
        self.bus.send_message(msg)
        
        received = self.bus.receive_message("B")
        self.assertIsNotNone(received)
        self.assertEqual(received.sender, "A")


class TestAgents(unittest.TestCase):
    """Tests des agents"""
    
    def setUp(self):
        self.chef = ChefProjetAgent()
        self.engineer = DataEngineerAgent()
    
    def tearDown(self):
        self.chef.stop()
        self.engineer.stop()
    
    def test_creation(self):
        """Test de création"""
        self.assertEqual(self.chef.name, "ChefProjet")
        self.assertEqual(self.engineer.name, "DataEngineer")
    
    def test_start_stop(self):
        """Test démarrage/arrêt"""
        self.chef.start()
        self.assertTrue(self.chef.is_running)
        self.chef.stop()
        self.assertFalse(self.chef.is_running)


def run_tests():
    """Lance les tests"""
    print("\n" + "="*50)
    print("Tests du système multi-agent")
    print("="*50 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestMessageBus))
    suite.addTests(loader.loadTestsFromTestCase(TestAgents))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*50)
    print("Résumé")
    print("="*50)
    print(f"Tests réussis: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    print("="*50 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
