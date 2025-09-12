# Archivo: main.py

import pandas as pd
import numpy as np
import warnings

# Asegúrate de que los nombres de tus archivos y clases sean correctos
from LogisticGD import ModeloLogistico
from Stocashtic import ModeloSGD 
from MiniBach import ModeloMiniBatch 
from TreeModel import TreeModel
from SvmModel import SVMclass

# Ignorar advertencias de scikit-learn para una interfaz más limpia
warnings.filterwarnings("ignore", category=UserWarning)


def obtener_datos_usuario_completo():
    print("\n--- Por favor, ingresa los datos del paciente ---")

    # Nombres de las columnas en el orden exacto del CSV original
    preguntas = {
        'GenHlth': "Salud general (escala 1-5)?",
        'BMI': "Índice de Masa Corporal (BMI)?",
        'Age': "Categoría de edad (escala 1-13)?",
        'HighBP': "Presión arterial alta (1=Sí, 0=No)?",
        'HighChol': "Colesterol alto (1=Sí, 0=No)?",
        'DiffWalk': "Dificultad para caminar (1=Sí, 0=No)?",
        'HeartDiseaseorAttack': "Enfermedad cardíaca o ataque (1=Sí, 0=No)?",
        'PhysHlth': "Días de mala salud física en el último mes?",
        'Stroke': "¿Ha tenido un derrame cerebral (1=Sí, 0=No)?",
        'CholCheck': "¿Ha tenido un chequeo de colesterol en los últimos 5 años (1=Sí, 0=No)?",
        'MentHlth': "Días de mala salud mental en el último mes?",
        
        
        
        
    }
    columnas = ['GenHlth', 'BMI', 'Age', 'HighBP', 'HighChol', 'DiffWalk', 'HeartDiseaseorAttack', 'PhysHlth', 'Stroke', 'CholCheck', 'MentHlth']

    datos_usuario = []
    for col in columnas:
        while True:
            try:
                valor = float(input(f"- {preguntas[col]}: "))
                datos_usuario.append(valor)
                break
            except ValueError:
                print("Error: Por favor, ingresa un número válido.")
    return np.array(datos_usuario).reshape(1, -1)


def main():
    # Ruta al archivo CSV. Asegúrate de que la ruta sea correcta.
    file_path = "../data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

    modelos_disponibles = {
        "1": ModeloLogistico,
        "2": ModeloSGD,
        "3": ModeloMiniBatch,
        "4": TreeModel,
        "5": SVMclass,
        
    }

    while True:
        print("\n" + "="*50)
        print("MENÚ DE PREDICCIÓN DE RIESGO DE DIABETES")
        print("="*50)
        print("Elige el modelo que deseas usar:")
        print("   1: Regresión Logística (Recomendado)")
        print("   2: SGD (Stochastic Gradient Descent)")
        print("   3: Mini-Batch Gradient Descent")
        print("   4: Árbol de Decisión")
        print("   5: SVM")
        opcion = input("Ingresa tu opción (o 'salir' para terminar): ")

        if opcion.lower() == 'salir':
            break

        if opcion not in modelos_disponibles:
            print("\n⚠️ Opción no válida. Inténtalo de nuevo.")
            continue

        try:
            print('cargando....')
            # 1. Instanciar el modelo (carga el dataset completo)
            ClaseDelModelo = modelos_disponibles[opcion]
            modelo = ClaseDelModelo(file_path)
            # 2. **Paso Clave:** Modificamos el objeto DESPUÉS de crearlo.
            #    Filtramos el atributo _X para que solo contenga las 7 columnas importantes.
            columnas_importantes = [
                'GenHlth', 'BMI', 'Age', 'HighBP', 'HighChol', 'DiffWalk', 'HeartDiseaseorAttack', 'PhysHlth', 'Stroke', 'CholCheck', 'MentHlth'
            ]
            modelo._X = modelo.X[columnas_importantes]

            # 2. Preparar los datos
            modelo.split()
            modelo.scale()

            # 3. Entrenar el modelo
            modelo.train()

            # 4. Mostrar el rendimiento del modelo
            modelo.metrics()

            # 5. Pedir datos al usuario
            datos_usuario = obtener_datos_usuario_completo()

            # 6. Realizar y mostrar la predicción
            prediccion, probabilidades = modelo.predict(datos_usuario)
            
            print("\n--- Resultado de la Predicción ---")
            if prediccion[0] == 1.0:
                if probabilidades:
                    confianza = probabilidades[0][1] * 100
                    print(f"Resultado: ALTO RIESGO de diabetes.")
                    print(f"Confianza de la predicción: {confianza:.2f}%")
            else:
                if probabilidades:
                    confianza = probabilidades[0][0] * 100
                    print(f"Resultado: BAJO RIESGO de diabetes.")
                    print(f"Probabilidad de que tengas diabetes: {confianza:.2f}%")

        except FileNotFoundError:
            print(f"Error: No se encontró el archivo en la ruta '{file_path}'. Verifica que el archivo exista.")
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}")

        input("\n--- Presiona Enter para volver al menú ---")


if __name__ == "__main__":
    main()