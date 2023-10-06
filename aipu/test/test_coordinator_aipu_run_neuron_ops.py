import unittest
from aipu.aipu import coordinator_aipu_processor

class neuron_ops_test_cases(unittest.TestCase):

    def test_run_neuron_ops(self):
        expected_result = "Successful"
        test_aipu = coordinator_aipu_processor(
            aipu_id = "1",
            pesos = [1,2],
            biases = [2],
            fas = "ReLu",
            capa_id = "1",
            output_name = "o1",
            output_ip = "192.168.0.1",
            output_port = "6339",
            input_name = "i1"
        )

        if test_aipu.set_inputs([2,4], "i1"):
            result = test_aipu.run_neuron_ops()

            self.assertEqual(expected_result, result)

if __name__ == '__main__':
    unittest.main()
