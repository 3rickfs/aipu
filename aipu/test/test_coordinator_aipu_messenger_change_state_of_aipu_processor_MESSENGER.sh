echo "************************************************************************************"
echo "Test 1: Mensajero coordinador cambia el estado del procesador coordinador cambiando el valor de una variable booleana"
echo "Coordinator messenger aipu"

source ~/dev/edge-intelligence-simulator/env/bin/activate

python ~/dev/edge-intelligence-simulator/aipu/aipu/aipu-messenger-coordinator.py 127.0.0.1 65432 1

echo "************************************************************************************"

