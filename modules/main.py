import pandas as pd
import numpy as np
from LogisticGD import ModeloLogistico
from MiniBach import  ModeloMiniBatch
from Stocashtic import ModeloSGD

def obtener_datos_usuario():
    print("\n--- Por favor, ingresa tus datos ---")
    preguntas = {
        'GenHlth': "Salud general (escala 1-5)?",
        'BMI': "Índice de Masa Corporal (BMI)?",
        'Age': "Categoría de edad (escala 1-13)?",
        'HighBP': "Presión arterial alta (1=Sí, 0=No)?",
        'HighChol': "Colesterol alto (1=Sí, 0=No)?",
        'DiffWalk': "Dificultad para caminar (1=Sí, 0=No)?",
        'HeartDiseaseorAttack': "Enfermedad cardíaca o ataque (1=Sí, 0=No)?"
    }
    columnas = ['GenHlth', 'BMI', 'Age', 'HighBP', 'HighChol', 'DiffWalk', 'HeartDiseaseorAttack']
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

    file = "../data/diabetes_binary_health_indicators_BRFSS2015.csv"

    modelos = {
        "1": ModeloLogistico,
        "2": ModeloMiniBatch,
        "3": ModeloSGD,
        
    }

    while True:
        print("\n" + "="*50)
        print("MENÚ DE PREDICCIÓN DE RIESGO DE DIABETES")
        print("="*50)
        print("Elige el modelo que deseas usar:")
        print("   1: Regresión Logística")
        print("   2: Mini-Batch SGD")
        print("   3: SGD")
        
        opcion = input("Ingresa tu opción (o 'salir' para terminar): ")

        if opcion.lower() == 'salir':
            break
        
        if opcion not in modelos:
            print("\n Opción no válida. Inténtalo de nuevo.")
            continue
        
        # --- Flujo de trabajo ---
        # 1. Instanciar el modelo elegido
        ClaseDelModelo = modelos[opcion]
        modelo = ClaseDelModelo(file)
        
        # 2. Preparar los datos
        modelo.split()
        modelo.scale()
        
        # 3. Entrenar el modelo
        modelo.train()
        
        # 4. Mostrar el rendimiento del modelo recién entrenado
        modelo.metrics()
        
        # 5. Pedir datos al usuario para una predicción en vivo
        datos_usuario = obtener_datos_usuario()
        
        # 6. Realizar y mostrar la predicción
        prediccion, probabilidades = modelo.predict_user_data(datos_usuario)
        
        print("\n--- Resultado de la Predicción ---")
        if prediccion[0] == 1.0:
            confianza = probabilidades[0][1] * 100
            print(f"Resultado: ALTO RIESGO de diabetes.")
            print(f"Confianza de la predicción: {confianza:.2f}%")
        else:
            confianza = probabilidades[0][0] * 100
            print(f"Resultado: BAJO RIESGO de diabetes.")
            print(f"Confianza de la predicción: {confianza:.2f}%")
        
        input("\n--- Presiona Enter para volver al menú ---")

if __name__ == "__main__":
    main()