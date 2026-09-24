"""Plataforma de OCR y ampliación de texto con OpenAI y Streamlit."""

from __future__ import annotations

import io
import base64
import hashlib
import json
import re
import shutil
import time
from collections import Counter

import pandas as pd
import pytesseract
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageOps


st.set_page_config(page_title="OCR + LLM", page_icon="🧾", layout="wide")
st.title("🧾 OCR + ampliación con LLM")
st.caption("Extrae texto de una imagen, amplíalo con OpenAI y revisa métricas lingüísticas.")


def metricas_basicas(texto: str) -> dict[str, int | float]:
    palabras = re.findall(r"\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b", texto, flags=re.UNICODE)
    oraciones = [s for s in re.split(r"(?<=[.!?])\s+", texto.strip()) if s]
    parrafos = [p for p in re.split(r"\n\s*\n", texto.strip()) if p.strip()]
    conteo = Counter(p.lower() for p in palabras)
    unicas = len(conteo)
    total = len(palabras)
    return {
        "Caracteres": len(texto),
        "Palabras": total,
        "Oraciones": len(oraciones),
        "Párrafos": len(parrafos),
        "Palabras únicas": unicas,
        "Diversidad léxica (%)": round(100 * unicas / total, 1) if total else 0,
        "Promedio de palabras por oración": round(total / len(oraciones), 1) if oraciones else 0,
    }


def evaluar_texto(client: OpenAI, model: str, texto: str, temperature: float) -> dict:
    prompt = """Evalúa el siguiente texto en español. Devuelve solo JSON válido, sin markdown, con estas claves:
coherencia, semantica, sintaxis, gramatica (enteros de 0 a 100), observaciones (lista de hasta 3 hallazgos concretos).
Usa una escala donde 100 es excelente. Evalúa únicamente el texto entregado, sin inferir hechos externos.
Texto:\n""" + texto[:12000]
    result = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Eres un evaluador lingüístico cuidadoso. Sé consistente y constructivo."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    data = json.loads(result.choices[0].message.content or "{}")
    for field in ("coherencia", "semantica", "sintaxis", "gramatica"):
        data[field] = max(0, min(100, int(data.get(field, 0))))
    data["observaciones"] = data.get("observaciones", [])
    return data


with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-…", help="Se conserva solo en la sesión actual.")
    model = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)
    temperature = st.slider("Temperatura", 0.0, 2.0, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.05, 1.0, 1.0, 0.05)
    max_tokens = st.slider("Máximo de tokens", 128, 8192, 1200, 128)
    estilo = st.radio("Estilo de respuesta", ["Formal", "Técnico"], horizontal=True)
    idioma_ocr = st.selectbox("Idioma del OCR", ["Español", "Inglés", "Español + inglés"])

col_img, col_ocr = st.columns([1, 1])
with col_img:
    st.subheader("1. Carga una imagen")
    archivo = st.file_uploader("PNG, JPG o JPEG", type=["png", "jpg", "jpeg"])
    if archivo:
        try:
            imagen = Image.open(io.BytesIO(archivo.getvalue()))
            st.image(imagen, caption=archivo.name, use_container_width=True)
        except Exception as exc:
            st.error(f"No se pudo abrir la imagen: {exc}")

with col_ocr:
    st.subheader("2. Texto reconocido")
    if archivo:
        image_id = hashlib.sha256(archivo.getvalue()).hexdigest()[:16]
        ocr_key = f"ocr_text_{image_id}"
        if ocr_key not in st.session_state:
            if shutil.which("tesseract") is not None:
                lang = {"Español": "spa", "Inglés": "eng", "Español + inglés": "spa+eng"}[idioma_ocr]
                try:
                    st.session_state[ocr_key] = pytesseract.image_to_string(
                        ImageOps.grayscale(imagen), lang=lang
                    ).strip()
                except Exception as exc:
                    st.warning(f"Tesseract no pudo leer esta imagen ({exc}). Puedes probar OpenAI Vision.")
                    st.session_state[ocr_key] = ""
            else:
                st.info("Tesseract no está instalado. Puedes instalarlo o extraer el texto con OpenAI Vision.")
                st.session_state[ocr_key] = ""
        if not st.session_state[ocr_key] and st.button(
            "Extraer texto con OpenAI Vision", disabled=not api_key.strip(), key=f"vision_{image_id}"
        ):
            try:
                mime = archivo.type or "image/png"
                encoded = base64.b64encode(archivo.getvalue()).decode("ascii")
                vision_client = OpenAI(api_key=api_key.strip())
                with st.spinner("Reconociendo el texto de la imagen con OpenAI Vision…"):
                    vision_result = vision_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Transcribe exactamente todo el texto visible en esta imagen. Conserva el idioma {idioma_ocr}, los párrafos y la puntuación. Devuelve solo la transcripción, sin comentarios."},
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                            ],
                        }],
                        max_tokens=2000,
                        temperature=0,
                    )
                st.session_state[ocr_key] = vision_result.choices[0].message.content or ""
                st.rerun()
            except Exception as exc:
                st.error(f"No se pudo extraer el texto con OpenAI Vision: {exc}")
        st.text_area("OCR editable", height=260, key=ocr_key)
        texto_ocr = st.session_state[ocr_key]
    else:
        st.info("Carga una imagen para iniciar el reconocimiento.")
        texto_ocr = ""

st.subheader("3. Instrucción para ampliar")
instruccion = st.text_area(
    "¿Qué debe hacer el modelo con el texto?",
    value="Amplía y explica las ideas principales, conservando el significado del texto original.",
    height=90,
)

if st.button("Analizar y ampliar", type="primary", disabled=not (archivo and texto_ocr.strip() and api_key.strip())):
    try:
        client = OpenAI(api_key=api_key.strip())
        role_style = "formal, preciso y profesional" if estilo == "Formal" else "técnico, preciso y con terminología especializada"
        user_prompt = (
            f"Estilo requerido: {role_style}.\nInstrucción: {instruccion.strip()}\n\n"
            "Texto extraído por OCR (puede contener errores; corrige solo los evidentes sin cambiar el sentido):\n"
            f"{texto_ocr[:20000]}"
        )
        started = time.perf_counter()
        with st.spinner("Generando la respuesta…"):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Responde en español. Amplía el contenido con claridad y mantén fidelidad al texto fuente."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message.content or ""
            usage = response.usage
            quality = evaluar_texto(client, model, answer, temperature)
        st.session_state.analysis = {
            "answer": answer,
            "ocr": texto_ocr,
            "quality": quality,
            "usage": usage,
            "latency": time.perf_counter() - started,
            "model": model,
            "style": estilo,
        }
    except Exception as exc:
        st.error(f"No se pudo completar el análisis: {exc}")

if "analysis" in st.session_state:
    result = st.session_state.analysis
    st.divider()
    st.header("Resultado")
    with st.container(border=True):
        st.markdown(result["answer"])
    usage = result["usage"]
    a, b, c, d = st.columns(4)
    a.metric("Tokens de entrada", usage.prompt_tokens if usage else "—")
    b.metric("Tokens de salida", usage.completion_tokens if usage else "—")
    c.metric("Latencia total", f"{result['latency']:.2f} s")
    d.metric("Modelo / estilo", f"{result['model']} · {result['style']}")

    st.subheader("Métricas lingüísticas")
    st.caption("Las puntuaciones lingüísticas son una evaluación orientativa del LLM, no una medición objetiva ni una certificación.")
    quality = result["quality"]
    score_cols = st.columns(4)
    for column, label, key in zip(score_cols, ["Coherencia", "Semántica", "Sintaxis", "Gramática"], ["coherencia", "semantica", "sintaxis", "gramatica"]):
        column.metric(label, f"{quality[key]}/100")
    if quality.get("observaciones"):
        st.markdown("**Observaciones**")
        for note in quality["observaciones"]:
            st.write(f"- {note}")
    st.subheader("Medidas del texto generado")
    stats = metricas_basicas(result["answer"])
    st.dataframe(pd.DataFrame([stats]), hide_index=True, use_container_width=True)
    with st.expander("Ver texto OCR utilizado"):
        st.text(result["ocr"])
