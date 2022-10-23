# text_clasification

IA con tensorflow  python para clasificar textos

Para poder trabajar con con este servidor es importante realizar los siguientes pasos:

1) Clonar el repo

2) Instalar las librerias necesarias con $ pip install -r requirements.text
en caso de que que esta libreria "certifi @ file:///C:/b/abs_ac29jvt43w/croot/certifi_1665076682579/work/certifi" provoque un error , simplemente podemos eliminarla del archivo y volver a ejecutar el comando de instalación antes mencionado

2) Dado que se creo un enviroment usando conda en este proyecto (llamado envclasification), es importante verificar que tengamos instalado conda.
(si se usa linux y no se reconce conda, podemos instalar miniconda)

3) Una vez verificada la instalacion de conda , procedemos a ejecutar el comando $conda activate envclasification para ingresar la environment 

4) Este proyecto utiliza UVICORN con FastAPI para ejecutar un servidor 
Por lo tanto deberemos ejecutar $uvicorn main:app --reload


5) el proceso puede correr en local usando el puerto :8000 , es decir 127.0.0.1:8000

6) Al tipear la URL en la barra de busqueda , el resultado debe ser {"test: "test} , lo que indicará que el proceso funciona perfecto

7)Por ultimo para probar la funcionalidad , ser debe ir a 127.0.0.1:8000/predictor/<Direccion de la pagina web>
el resultado debera ser un numero flotante entre 0 y 1
  
  
  

# IMPORTANT !!

# To activate this environment, use

# $ conda activate envclasification

# To deactivate an active environment, use

# $ conda deactivate



# $ uvicorn <nombre_de_archivo>:app --reload
