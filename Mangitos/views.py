import base64

from django.shortcuts import render

from src.predict import predecir_imagen

LIMITE_MB = 5
CONFIANZA_MINIMA = 60


def home(request):
    return render(request, "home.html")


def predecir(request):
    contexto = {}

    if request.method == "POST":
        archivo = request.FILES.get("imagen")

        if not archivo:
            contexto["error"] = "Selecciona una imagen primero."
        elif archivo.size > LIMITE_MB * 1024 * 1024:
            contexto["error"] = f"La imagen pesa más de {LIMITE_MB} MB."
        else:
            try:
                datos = archivo.read()
                archivo.seek(0)
                contexto["imagen_b64"] = base64.b64encode(datos).decode()
                contexto["tipo_imagen"] = archivo.content_type

                resultado = predecir_imagen(archivo)
                contexto["resultado"] = resultado

                if resultado["confianza"] < CONFIANZA_MINIMA:
                    contexto["aviso"] = "El modelo no está seguro. Prueba con una foto más clara y con buena luz."
            except Exception as e:
                print("Error al predecir:", e)
                contexto["error"] = "No se pudo procesar la imagen. Prueba con un JPG o PNG."

    return render(request, "predecir.html", contexto)