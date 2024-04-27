#-------------------------------------------#
#   Nombre y Apellido = Mabrouka Salmi      #
#-------------------------------------------#


from mqtt_as import MQTTClient, config  # Importar las clases necesarias
import uasyncio as asyncio   # Importar la biblioteca asyncio para programación asíncrona
from random import randint  # Importar la función randint para generar números aleatorios

# Ts = 5
async def messages(client):
     # Recibir mensajes
    async for topic, msg, retained in client.queue:
        print(f'[CAPA APLICACIÓN] Topic: "{topic.decode()}" Message: "{msg.decode()}" Retained: {retained}')
        """
        global Ts
        if topic.decode() == 'T':
            Ts = int(msg.decode())
        elif topic.decode() == 'F':
            pass
            
        """

async def down(client):
    # Manejar la desconexión del cliente MQTT
    while True:
        await client.down.wait()  # Esperar a que la conexión se cierre
        client.down.clear()
        print('[CAPA COMUNICACIÓN] Conexion MQTT cerrada')  # Mensaje de desconexión

async def up(client):
    # Manejar la conexión exitosa del cliente MQTT
    while True:
        await client.up.wait()  # Esperar a que la conexión se establezca
        client.up.clear()
        print('[CAPA COMUNICACIÓN] Conexion MQTT establecida')   # Mensaje de conexión establecida
        
        # (re) suscripciones (tras evento de conexión o reconexión)
        for s in SUBS: 
            await client.subscribe(s, 1)     # Suscribirse a los temas

async def sensor(client):
    # Simular un sensor que publica datos aleatorios cada 2 segundos (Ts)
    print('[CAPA PERCEPCIÓN] Sensor iniciado ...')
    while True:
        await asyncio.sleep(2)  # Esperar 2 segundos (Ts)
        v = randint(0,100)  # Generar un valor aleatorio
        print(f'[CAPA PERCEPCION] v=%d'%(v))    # Imprimir el valor
        await client.publish('v', '%d'%v, qos = 1)  # Publicar el valor como mensaje 'v'


async def main(client):
    try:
        print('[CAPA COMUNICACIÓN] Iniciando conexion...')  # Mensaje de inicio de conexión
        await client.connect()   # Intentar conectar al servidor MQTT
        print('[CAPA COMUNICACIÓN] Conexión establecida')   # Mensaje de conexión exitosa
    except OSError:
        print('[CAPA COMUNICACIÓN] Conexión fallida')   # Mensaje de conexión fallida
        return
    for task in (up, down, messages):
        asyncio.create_task(task(client))   # Crear tareas para manejar conexión/desconexión y mensajes
        
    # crear aquí las tareas que sean necesarias
    await sensor(client)    # Ejecutar la simulación del sensor 
    
if __name__ == '__main__':
    
    # Configuración de conexión MQTT
    config['server'] = '192.168.2.5'    # Dirección del servidor MQTT
    config['ssid'] = 'IOTNET_2.4'    # SSID de la red WiFi
    config['wifi_pw'] = '10T@ATC_'  # Contraseña de la red WiFi
    config["user"]= ""  # Usuario MQTT (si es necesario)
    config["password"]= ""  # Contraseña MQTT (si es necesaria)
    config['keepalive'] = 120    # Intervalo de keepalive para la conexión
    config["queue_len"]= 5  # Longitud de la cola de mensajes
    config['will'] = ('topic_final', 'Mensaje de finalizacion', False, 0)   # Mensaje de desconexión

    # suscripciones: Lista de temas a los que suscribirse
    SUBS = ('topic_i1', 'topic_i2')

    # configuración y creación de la clase cliente
    MQTTClient.DEBUG = True # Activar modo de depuración
    client = MQTTClient(config) # Crear instancia del cliente MQTT

    # ejecución de la rutina main
    try:
        asyncio.run(main(client))   # Ejecutar la rutina principal con la instancia del cliente
    finally:
        client.close()  # Cerrar la conexión MQTT al finalizar
        asyncio.new_event_loop()    # Crear un nuevo bucle de eventos asyncio
