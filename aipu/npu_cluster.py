import os
from abc import ABC, abstractmethod
import json

class neuron_ops(ABC):
    """
    """

    @abstractmethod
    def run_operation(**kwargs):
        #interface to child classes
        pass

class operation_1(neuron_ops):
    """ run operation 1
    """

    def run_operation(**kwargs):
        print("running operation 1")
        kwargs["result"] = "Successful"

        return kwargs

class NPUClusterOps:

    @staticmethod
    def run(**kwargs):
        for operation in neuron_ops.__subclasses__():
            kwargs = operation.run_operation(kwargs)

        return kwargs["result"]










