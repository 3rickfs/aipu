# Create the aipu classes
from . npu_cluster import NPUClusterOps

class aipu():
    """ Define el contructor de un AIPU
    """

    def __init__(self, aipu_id):
        self.aipu_id = aipu_id
        self.version = "1.0.0"
        self.status = ""

class coordinator_aipu_processor(aipu):
    """ Define el aipu procesador coordinador incluyendo sus atributos y metodos
    """

    def __init__(
        self,
        aipu_id,
        pesos,
        biases,
        fas,
        capa_id,
        output_name,
        output_ip,
        output_port,
        input_name
    ):
        super().__init__(aipu_id)
        self.pesos = pesos
        self.biases = biases
        self.fas = fas
        self.capa_id = capa_id
        self.output_name = output_name
        self.output_ip = output_ip
        self.output_port = output_port
        self.input_name = input_name
        self.inputs = []

        #self.instrucciones = self.get_format_instr(instrucciones) 
        #self.procesadores = self.get_format_proc(procesadores)

    def run_neuron_ops(self):
        print("Running the neurons, muchachon!")
        try:
            result = NPUClusterOps.run(
                pesos = self.pesos,
                biases = self.bias,
                fas = self.fas,
                output_name = self.output_name,
                output_ip = self.output_ip,
                output_port = self.output_port
            )
        except Exception as e:
            result = {
                'error': e
            }

        return result

    def verify_input(self, entrante_input_name):
        if self.input_name == entrante_input_name:
            return True
        else:
            return False

    def set_inputs(self, inputs, entrante_input_name):
        if self.verify_input(entrante_input_name):
            print("Inputs accepted")
            self.inputs = inputs
        else:
            print("Inputs rejected")


