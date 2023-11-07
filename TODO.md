# TODO

- Desarrollar un test donde un AIPU-M-W le cambie el estado a su AIPU-P-W
- Desarrollar un test donde el AIPU-M-C le cambie el estado a un AIPU-P-W
- Desarrollar un test donde un AIPU-M-W le diga a su AIPU-P-W que le devuelva un mensaje cuando un AIPU-M-C le cambie el estado.
- Desarrollar un test en el que el AIPU-M-W luego de ser cambiado su estado por su AIPU-P-W, le cambie el estado a otro AIPU-P-W. Este test asegura la comunicacion consecutiva entre aipus, a lo cual se le denominaria una rama. 
- Desarrollar un test en el que se realice los test de arriba pero para dos ramas. Asegurar que mediante este test, esto sea aplicado a n ramas. Se debe considerar una especie de arreglo especifico para que se guarden las direcciones y puerto de cada aipu. 

- Crear un test donde se establezca manualmente las instrucciones correspondientes del modelo. Importante incluir la clase aipu en el processor y messenger coordinators.
- Desarrollar un test donde se convierta una red MLP hecha con TF a un JSON con instrucciones. 
- Test para que ingrese un modelo MLP hecho con TF a un AIPU-M-C y este lo comunique a los aipus workers correspondientes, y ase ejecute las predicciones, devolviendo el resultado.

- Test para comunicar entre containers/pods, abriendo o habilitando la red correspondiente para que puedan escucharse.
- Test de cominicacion entre containers/pods en diferentes lugares conectados a traves de una red VPN.

