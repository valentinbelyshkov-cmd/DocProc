import unittest
import json
from unittest.mock import patch, MagicMock
from app import app, allowed_file

class TestOCRApp(unittest.TestCase):
    
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()
    
    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_allowed_file(self):
        self.assertTrue(allowed_file('test.pdf'))
        self.assertTrue(allowed_file('test.png'))
        self.assertFalse(allowed_file('test.txt'))
        self.assertFalse(allowed_file('test'))
        self.assertFalse(allowed_file(''))

    @patch('requests.get')
    def test_task_status_failed_api(self, mock_get):
        # Mock API failure
        mock_get.side_effect = Exception("API Down")
        
        response = self.app.get('/api/task_status/test_id')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "failed")
        self.assertIn("API Down", data["error"])

if __name__ == '__main__':
    unittest.main()
