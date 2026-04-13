import io
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Procesador MIC MAC ajustado",
    layout="wide"
)

# =========================================================
# CONFIG VISUAL
# =========================================================
COLOR_AZUL = "#1F77B4"
COLOR_VERDE = "#2CA02C"
COLOR_NARANJA = "#FF7F0E"
COLOR_ROJO = "#D62728"
COLOR_GRIS = "#7F7F7F"
COLOR_MORADO = "#7E57C2"

COLOR_CUAD_CONFLICTO = "#FDECEC"
COLOR_CUAD_PODER = "#ECF7EC"
COLOR_CUAD_RESULTADO = "#EAF3FB"
COLOR_CUAD_AUTONOMA = "#FFF3E8"
COLOR_CUAD_FRONTERA = "#F3ECFD"

TOLERANCIA_DEFAULT = 2.0


# =========================================================
# HELPERS GENERALES
# =========================================================
def limpiar_texto(valor):
    if pd.isna(valor):
        return ""
    txt = str(valor).strip()
    txt = re.sub(r"\s+", " ", txt)
    return txt


def es_vacio(valor):
    txt = limpiar_texto(valor)
    return txt == "" or txt.lower() == "nan"


def normalizar_codigo(valor):
    txt = limpiar_texto(valor).upper()
    txt = txt.replace("\n", " ")
    txt = re.sub(r"\s+", "", txt) if len(txt) <= 20 else txt
    return txt


def nombre_archivo_sin_extension(nombre):
    return Path(nombre).stem


def color_zona(zona):
    if zona == "PODER":
        return COLOR_VERDE
    if zona == "CONFLICTO":
        return COLOR_ROJO
    if zona == "RESULTADO":
        return COLOR_AZUL
    if zona == "AUTONOMA":
        return COLOR_NARANJA
    if zona == "FRONTERA":
        return COLOR_MORADO
    return COLOR_GRIS


# =========================================================
# DETECCIÓN DE HOJAS
# =========================================================
def puntuar_hoja_descriptores(xls, hoja):
    try:
        df = pd.read_excel(xls, sheet_name=hoja, header=None, nrows=20)
    except Exception:
        return -999

    texto = " | ".join(
        [limpiar_texto(x).upper() for x in df.fillna("").astype(str).values.flatten().tolist()]
    )

    score = 0
    if "DESCRIPTOR" in texto:
        score += 5
    if "NOMBRE CORTO" in texto:
        score += 5
    if "CATEGOR" in texto:
        score += 2
    if "PROBLEM" in texto:
        score += 1
    if "MATRIZ" in hoja.upper():
        score -= 3

    return score


def detectar_hoja_descriptores(xls):
    mejores = [(hoja, puntuar_hoja_descriptores(xls, hoja)) for hoja in xls.sheet_names]
    mejores = sorted(mejores, key=lambda x: x[1], reverse=True)
    return mejores[0][0] if mejores else xls.sheet_names[0]


def puntuar_hoja_matriz(xls, hoja):
    try:
        df = pd.read_excel(xls, sheet_name=hoja, header=None, nrows=50)
    except Exception:
        return -999

    flat = [limpiar_texto(x).upper() for x in df.fillna("").astype(str).values.flatten().tolist()]
    texto = " | ".join(flat)

    score = 0
    if "MATRIZ" in hoja.upper():
        score += 6
    if "PARTICIPANTE" in texto:
        score += 2
    if "INSTITUCIÓN" in texto or "INSTITUCION" in texto:
        score += 2

    nums = sum(1 for x in flat if x in {"0", "1", "2", "3"})
    score += min(nums, 20) * 0.2
    return score


def detectar_hoja_matriz(xls):
    mejores = [(hoja, puntuar_hoja_matriz(xls, hoja)) for hoja in xls.sheet_names]
    mejores = sorted(mejores, key=lambda x: x[1], reverse=True)
    return mejores[0][0] if mejores else None


# =========================================================
# DESCRIPTORES
# =========================================================
def detectar_header_descriptores(df_raw):
    for i in range(min(len(df_raw), 25)):
        fila = [limpiar_texto(x).upper() for x in df_raw.iloc[i].tolist()]
        fila_txt = " | ".join(fila)
        if "NOMBRE CORTO" in fila_txt and "DESCRIPTOR" in fila_txt:
            return i
    return None


def leer_descriptores(xls):
    hoja = detectar_hoja_descriptores(xls)
    df_raw = pd.read_excel(xls, sheet_name=hoja, header=None)

    fila_header = detectar_header_descriptores(df_raw)
    if fila_header is None:
        raise ValueError(f"No se pudo detectar el encabezado de descriptores en la hoja '{hoja}'.")

    df = pd.read_excel(xls, sheet_name=hoja, header=fila_header)
    df.columns = [limpiar_texto(c).upper() for c in df.columns]

    col_codigo = None
    col_descriptor = None
    col_categoria = None

    for c in df.columns:
        if "NOMBRE CORTO" in c:
            col_codigo = c
        elif "DESCRIPTOR" in c:
            col_descriptor = c
        elif "CATEGOR" in c:
            col_categoria = c

    if not col_codigo or not col_descriptor:
        raise ValueError(
            f"La hoja '{hoja}' no contiene las columnas esperadas 'NOMBRE CORTO' y 'DESCRIPTOR'."
        )

    base = df[[col_codigo, col_descriptor]].copy()
    base["CATEGORIA"] = df[col_categoria] if col_categoria else ""

    base.columns = ["CODIGO", "DESCRIPTOR", "CATEGORIA"]
    base["CODIGO"] = base["CODIGO"].apply(normalizar_codigo)
    base["DESCRIPTOR"] = base["DESCRIPTOR"].apply(limpiar_texto)
    base["CATEGORIA"] = base["CATEGORIA"].apply(limpiar_texto)

    base = base[
        (base["CODIGO"] != "") &
        (base["DESCRIPTOR"] != "")
    ].drop_duplicates(subset=["CODIGO"], keep="first").reset_index(drop=True)

    if base.empty:
        raise ValueError(f"No se encontraron descriptores válidos en la hoja '{hoja}'.")

    return base, hoja


# =========================================================
# PARTICIPANTES / INSTITUCIONES
# =========================================================
def detectar_fila_participantes(df_raw):
    for i in range(min(len(df_raw), 50)):
        fila = [limpiar_texto(x).lower() for x in df_raw.iloc[i].tolist()]
        texto = " | ".join(fila)
        if "participante" in texto and ("institución" in texto or "institucion" in texto):
            return i
    return None


def detectar_columnas_participantes(header_row):
    cols = [limpiar_texto(x).lower() for x in header_row]
    col_part = None
    col_inst = None
    col_puesto = None

    for idx, c in enumerate(cols):
        if "participante" in c and col_part is None:
            col_part = idx
        if ("institución" in c or "institucion" in c) and col_inst is None:
            col_inst = idx
        if ("puesto" in c or "cargo" in c or "función" in c or "funcion" in c) and col_puesto is None:
            col_puesto = idx

    return col_part, col_inst, col_puesto


def leer_participantes_instituciones(df_raw):
    fila_header = detectar_fila_participantes(df_raw)

    if fila_header is None:
        return (
            pd.DataFrame(columns=["PARTICIPANTE", "INSTITUCION", "PUESTO"]),
            pd.DataFrame(columns=["INSTITUCION"])
        )

    header = df_raw.iloc[fila_header].tolist()
    col_part, col_inst, col_puesto = detectar_columnas_participantes(header)

    if col_part is None and col_inst is None:
        return (
            pd.DataFrame(columns=["PARTICIPANTE", "INSTITUCION", "PUESTO"]),
            pd.DataFrame(columns=["INSTITUCION"])
        )

    participantes = []
    i = fila_header + 1

    while i < len(df_raw):
        fila = df_raw.iloc[i].tolist()

        if all(es_vacio(v) for v in fila):
            break

        participante = limpiar_texto(fila[col_part]) if col_part is not None and col_part < len(fila) else ""
        institucion = limpiar_texto(fila[col_inst]) if col_inst is not None and col_inst < len(fila) else ""
        puesto = limpiar_texto(fila[col_puesto]) if col_puesto is not None and col_puesto < len(fila) else ""

        primera = limpiar_texto(fila[0]).upper() if len(fila) > 0 else ""
        if participante == "" and institucion == "" and puesto == "" and primera != "":
            break

        if participante or institucion or puesto:
            participantes.append({
                "PARTICIPANTE": participante,
                "INSTITUCION": institucion,
                "PUESTO": puesto
            })

        i += 1

    df_part = pd.DataFrame(participantes)

    if df_part.empty:
        df_inst = pd.DataFrame(columns=["INSTITUCION"])
    else:
        df_inst = df_part[["INSTITUCION"]].copy()
        df_inst["INSTITUCION"] = df_inst["INSTITUCION"].apply(limpiar_texto)
        df_inst = df_inst[df_inst["INSTITUCION"] != ""]
        df_inst = df_inst.drop_duplicates(subset=["INSTITUCION"], keep="first").reset_index(drop=True)

    return df_part, df_inst


# =========================================================
# MATRIZ
# =========================================================
def fila_parece_header_matriz(fila):
    vals = [limpiar_texto(x) for x in fila]
    if len(vals) < 4:
        return False

    primera = vals[0]
    resto = [normalizar_codigo(x) for x in vals[1:] if not es_vacio(x)]

    if not es_vacio(primera):
        return False
    if len(resto) < 3:
        return False

    cortos = sum(1 for x in resto if len(x) <= 20)
    return cortos >= max(3, int(len(resto) * 0.7))


def detectar_fila_encabezado_matriz(df_raw):
    for i in range(min(len(df_raw), 100)):
        if fila_parece_header_matriz(df_raw.iloc[i].tolist()):
            return i
    return None


def leer_matriz_micmac(df_raw, catalogo_descriptores):
    fila_header = detectar_fila_encabezado_matriz(df_raw)

    if fila_header is None:
        raise ValueError("No se pudo detectar la fila de encabezados de la matriz MIC MAC.")

    encabezados = [normalizar_codigo(x) for x in df_raw.iloc[fila_header].tolist()]
    codigos_columnas = [c for c in encabezados[1:] if c != ""]

    if not codigos_columnas:
        raise ValueError("No se detectaron códigos de columnas en la matriz MIC MAC.")

    filas_matriz = []
    i = fila_header + 1

    while i < len(df_raw):
        fila = df_raw.iloc[i].tolist()

        if all(es_vacio(v) for v in fila):
            break

        codigo_fila = normalizar_codigo(fila[0]) if len(fila) > 0 else ""
        if codigo_fila == "":
            break

        valores = fila[1:1 + len(codigos_columnas)]

        fila_dict = {"CODIGO": codigo_fila}
        for j, cod_col in enumerate(codigos_columnas):
            valor = valores[j] if j < len(valores) else 0
            fila_dict[cod_col] = valor

        filas_matriz.append(fila_dict)
        i += 1

    if not filas_matriz:
        raise ValueError("No se encontraron filas válidas dentro de la matriz MIC MAC.")

    matriz_df = pd.DataFrame(filas_matriz)
    matriz_df["CODIGO"] = matriz_df["CODIGO"].apply(normalizar_codigo)

    for c in matriz_df.columns[1:]:
        matriz_df[c] = pd.to_numeric(matriz_df[c], errors="coerce").fillna(0).astype(int)
        matriz_df[c] = matriz_df[c].clip(0, 3)

    matriz_df = matriz_df.drop_duplicates(subset=["CODIGO"], keep="first").reset_index(drop=True)

    codigos_filas = matriz_df["CODIGO"].tolist()
    codigos_validos = [c for c in codigos_columnas if c in codigos_filas]

    if len(codigos_validos) < 2:
        raise ValueError("No se pudo construir una matriz cuadrada válida.")

    matriz_df = matriz_df[["CODIGO"] + codigos_validos].copy()
    matriz_df = matriz_df[matriz_df["CODIGO"].isin(codigos_validos)].copy()
    matriz_df = matriz_df.set_index("CODIGO")
    matriz_df = matriz_df.loc[codigos_validos, codigos_validos]

    for i in range(len(matriz_df)):
        matriz_df.iat[i, i] = 0

    map_desc = dict(zip(catalogo_descriptores["CODIGO"], catalogo_descriptores["DESCRIPTOR"]))
    map_cat = dict(zip(catalogo_descriptores["CODIGO"], catalogo_descriptores["CATEGORIA"]))

    analisis_base = pd.DataFrame({
        "CODIGO": matriz_df.index.tolist(),
        "DESCRIPTOR": [map_desc.get(c, c) for c in matriz_df.index.tolist()],
        "CATEGORIA": [map_cat.get(c, "") for c in matriz_df.index.tolist()]
    })

    return matriz_df, analisis_base


# =========================================================
# ANÁLISIS MICMAC AJUSTADO
# =========================================================
def clasificar_zona_micmac_real(inf, dep, media_inf, media_dep, tol):
    if inf == 0 and dep == 0:
        return "SIN RELACION"

    cerca_inf = abs(inf - media_inf) <= tol
    cerca_dep = abs(dep - media_dep) <= tol

    if cerca_inf or cerca_dep:
        return "FRONTERA"

    if inf > media_inf and dep < media_dep:
        return "PODER"

    if inf > media_inf and dep > media_dep:
        return "CONFLICTO"

    if inf < media_inf and dep > media_dep:
        return "RESULTADO"

    return "AUTONOMA"


def calcular_micmac(matriz_df, analisis_base, tolerancia=TOLERANCIA_DEFAULT):
    influencia = matriz_df.sum(axis=1)
    dependencia = matriz_df.sum(axis=0)

    df = analisis_base.copy()
    df["INFLUENCIA"] = df["CODIGO"].map(influencia.to_dict()).fillna(0).astype(int)
    df["DEPENDENCIA"] = df["CODIGO"].map(dependencia.to_dict()).fillna(0).astype(int)

    media_influencia = float(df["INFLUENCIA"].mean()) if not df.empty else 0.0
    media_dependencia = float(df["DEPENDENCIA"].mean()) if not df.empty else 0.0

    df["ZONA"] = df.apply(
        lambda row: clasificar_zona_micmac_real(
            row["INFLUENCIA"],
            row["DEPENDENCIA"],
            media_influencia,
            media_dependencia,
            tolerancia
        ),
        axis=1
    )

    orden_zona = {
        "PODER": 1,
        "CONFLICTO": 2,
        "RESULTADO": 3,
        "AUTONOMA": 4,
        "FRONTERA": 5,
        "SIN RELACION": 6
    }

    df["ORDEN_ZONA"] = df["ZONA"].map(orden_zona).fillna(99)
    df = df.sort_values(
        by=["ORDEN_ZONA", "INFLUENCIA", "DEPENDENCIA", "DESCRIPTOR"],
        ascending=[True, False, False, True]
    ).reset_index(drop=True)
    df = df.drop(columns=["ORDEN_ZONA"])

    return df, media_influencia, media_dependencia


# =========================================================
# VISUALES
# =========================================================
def generar_png_matriz(matriz_df):
    fig_w = max(10, len(matriz_df.columns) * 0.55)
    fig_h = max(8, len(matriz_df.index) * 0.50)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    arr = matriz_df.to_numpy()
    im = ax.imshow(arr, cmap="Blues", vmin=0, vmax=3)

    ax.set_xticks(np.arange(len(matriz_df.columns)))
    ax.set_yticks(np.arange(len(matriz_df.index)))
    ax.set_xticklabels(matriz_df.columns.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matriz_df.index.tolist(), fontsize=8)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = int(arr[i, j])
            ax.text(
                j, i, str(val),
                ha="center", va="center",
                color="white" if val >= 2 else "black",
                fontsize=6
            )

    ax.set_title("Matriz MIC MAC", fontsize=16, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def generar_png_mapa_micmac(df_micmac, media_influencia, media_dependencia, tolerancia):
    fig, ax = plt.subplots(figsize=(13, 8), dpi=180)

    max_dep = float(df_micmac["DEPENDENCIA"].max()) if not df_micmac.empty else 0
    max_inf = float(df_micmac["INFLUENCIA"].max()) if not df_micmac.empty else 0

    margen_x = max(media_dependencia, max_dep - media_dependencia) + 2
    margen_y = max(media_influencia, max_inf - media_influencia) + 2

    xmin = max(0, media_dependencia - margen_x)
    xmax = media_dependencia + margen_x
    ymin = max(0, media_influencia - margen_y)
    ymax = media_influencia + margen_y

    alto_total = ymax - ymin if ymax > ymin else 1
    ymin_ratio = (media_influencia - ymin) / alto_total

    ax.axvspan(xmin, media_dependencia, ymin=ymin_ratio, ymax=1, color=COLOR_CUAD_PODER, alpha=0.50, zorder=0)
    ax.axvspan(media_dependencia, xmax, ymin=ymin_ratio, ymax=1, color=COLOR_CUAD_CONFLICTO, alpha=0.50, zorder=0)
    ax.axvspan(media_dependencia, xmax, ymin=0, ymax=ymin_ratio, color=COLOR_CUAD_RESULTADO, alpha=0.50, zorder=0)
    ax.axvspan(xmin, media_dependencia, ymin=0, ymax=ymin_ratio, color=COLOR_CUAD_AUTONOMA, alpha=0.50, zorder=0)

    # Banda frontera
    ax.axvspan(media_dependencia - tolerancia, media_dependencia + tolerancia, color=COLOR_CUAD_FRONTERA, alpha=0.30, zorder=1)
    ax.axhspan(media_influencia - tolerancia, media_influencia + tolerancia, color=COLOR_CUAD_FRONTERA, alpha=0.30, zorder=1)

    ax.text(media_dependencia - (media_dependencia - xmin) * 0.55, media_influencia + (ymax - media_influencia) * 0.85,
            "PODER", fontsize=15, fontweight="bold", color="#1B5E20",
            ha="center", va="center", alpha=0.95)
    ax.text(media_dependencia + (xmax - media_dependencia) * 0.45, media_influencia + (ymax - media_influencia) * 0.85,
            "CONFLICTO", fontsize=15, fontweight="bold", color="#8B1E1E",
            ha="center", va="center", alpha=0.95)
    ax.text(media_dependencia + (xmax - media_dependencia) * 0.45, media_influencia - (media_influencia - ymin) * 0.85,
            "RESULTADO", fontsize=15, fontweight="bold", color="#0D47A1",
            ha="center", va="center", alpha=0.95)
    ax.text(media_dependencia - (media_dependencia - xmin) * 0.55, media_influencia - (media_influencia - ymin) * 0.85,
            "AUTÓNOMA", fontsize=15, fontweight="bold", color="#BF5A00",
            ha="center", va="center", alpha=0.95)

    ax.text(media_dependencia, ymax - 0.6, "FRONTERA", fontsize=11, fontweight="bold",
            color=COLOR_MORADO, ha="center", va="top", alpha=0.95)

    for _, row in df_micmac.iterrows():
        x = float(row["DEPENDENCIA"])
        y = float(row["INFLUENCIA"])
        z = row["ZONA"]
        codigo = row["CODIGO"]

        ax.scatter(
            x, y,
            s=220 if z != "FRONTERA" else 250,
            color=color_zona(z),
            edgecolors="black",
            linewidths=0.8,
            alpha=0.92,
            zorder=3
        )

        etiqueta_fija = f"{codigo:<11}"
        ax.annotate(
            etiqueta_fija,
            xy=(x, y),
            xytext=(x + 0.15, y + 0.18),
            textcoords="data",
            fontsize=7.5,
            ha="left",
            va="center",
            family="monospace",
            bbox=dict(
                boxstyle="square,pad=0.18",
                fc="#F7F2D8",
                ec="#8E8E8E",
                lw=0.8
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#6E6E6E",
                lw=0.7,
                shrinkA=0,
                shrinkB=0
            ),
            zorder=4
        )

    ax.axvline(media_dependencia, linestyle="--", color="#6F6F6F", linewidth=1.2, zorder=2)
    ax.axhline(media_influencia, linestyle="--", color="#6F6F6F", linewidth=1.2, zorder=2)

    ax.set_title("Mapa MIC MAC", fontsize=18, fontweight="bold")
    ax.set_xlabel("Dependencia", fontsize=13)
    ax.set_ylabel("Influencia", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


# =========================================================
# EXPORTACIÓN
# =========================================================
def exportar_excel_individual(
    nombre_archivo_fuente,
    participantes_df,
    instituciones_df,
    matriz_df,
    df_micmac,
    media_influencia,
    media_dependencia,
    tolerancia
):
    salida = io.BytesIO()

    with pd.ExcelWriter(salida, engine="xlsxwriter") as writer:
        resumen = pd.DataFrame({
            "CAMPO": [
                "Archivo fuente",
                "Fecha de procesamiento",
                "Cantidad de problemáticas",
                "Cantidad de participantes",
                "Cantidad de instituciones únicas",
                "Suma total influencia",
                "Suma total dependencia",
                "Media influencia",
                "Media dependencia",
                "Tolerancia",
                "Método"
            ],
            "VALOR": [
                nombre_archivo_fuente,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(df_micmac),
                len(participantes_df),
                len(instituciones_df),
                int(df_micmac["INFLUENCIA"].sum()) if not df_micmac.empty else 0,
                int(df_micmac["DEPENDENCIA"].sum()) if not df_micmac.empty else 0,
                media_influencia,
                media_dependencia,
                tolerancia,
                "MICMAC ajustado con frontera"
            ]
        })
        resumen.to_excel(writer, sheet_name="RESUMEN", index=False)
        participantes_df.to_excel(writer, sheet_name="PARTICIPANTES", index=False)
        instituciones_df.to_excel(writer, sheet_name="INSTITUCIONES", index=False)

        matriz_export = matriz_df.copy()
        matriz_export.insert(0, "CODIGO", matriz_export.index)
        matriz_export.to_excel(writer, sheet_name="MATRIZ_LIMPIA", index=False)

        df_micmac.to_excel(writer, sheet_name="ANALISIS_MICMAC", index=False)
        df_micmac[["ZONA", "CODIGO", "DESCRIPTOR", "CATEGORIA", "INFLUENCIA", "DEPENDENCIA"]].to_excel(
            writer, sheet_name="PROBLEMATICAS_POR_ZONA", index=False
        )

        hojas = {
            "RESUMEN": resumen,
            "PARTICIPANTES": participantes_df,
            "INSTITUCIONES": instituciones_df,
            "MATRIZ_LIMPIA": matriz_export,
            "ANALISIS_MICMAC": df_micmac,
            "PROBLEMATICAS_POR_ZONA": df_micmac[["ZONA", "CODIGO", "DESCRIPTOR", "CATEGORIA", "INFLUENCIA", "DEPENDENCIA"]]
        }

        for hoja, df in hojas.items():
            ws = writer.sheets[hoja]
            if df.empty:
                continue
            for idx, col in enumerate(df.columns):
                max_len = max(len(str(col)), df[col].astype(str).map(len).max() if len(df) > 0 else 0)
                ws.set_column(idx, idx, min(max(max_len + 2, 14), 55))

    return salida.getvalue()


# =========================================================
# PROCESAMIENTO CENTRAL
# =========================================================
def procesar_archivo_excel(file, tolerancia):
    xls = pd.ExcelFile(file)

    hoja_matriz = detectar_hoja_matriz(xls)
    if hoja_matriz is None:
        raise ValueError("No se encontró una hoja de matriz válida.")

    catalogo, hoja_desc = leer_descriptores(xls)
    df_raw_matriz = pd.read_excel(xls, sheet_name=hoja_matriz, header=None)

    participantes_df, instituciones_df = leer_participantes_instituciones(df_raw_matriz)
    matriz_df, analisis_base = leer_matriz_micmac(df_raw_matriz, catalogo)
    df_micmac, media_influencia, media_dependencia = calcular_micmac(
        matriz_df, analisis_base, tolerancia=tolerancia
    )

    return {
        "archivo": file.name,
        "hoja_descriptores": hoja_desc,
        "hoja_matriz": hoja_matriz,
        "catalogo": catalogo,
        "participantes": participantes_df,
        "instituciones": instituciones_df,
        "matriz": matriz_df,
        "analisis": df_micmac,
        "media_influencia": media_influencia,
        "media_dependencia": media_dependencia,
        "tolerancia": tolerancia
    }


# =========================================================
# INTERFAZ
# =========================================================
st.title("Procesador MIC MAC ajustado al software")
st.caption(
    "Sube uno o varios Excel. Esta versión usa tolerancia y zona frontera para acercarse más al comportamiento visual/analítico del software MICMAC."
)

tolerancia = st.number_input(
    "Tolerancia de frontera",
    min_value=0.0,
    max_value=10.0,
    value=TOLERANCIA_DEFAULT,
    step=0.5
)

archivos = st.file_uploader(
    "Carga uno o varios archivos MIC MAC",
    type=["xlsx"],
    accept_multiple_files=True
)

if archivos:
    resultados_ok = []
    errores = []

    progreso = st.progress(0, text="Procesando archivos...")
    total = len(archivos)

    for idx, archivo in enumerate(archivos, start=1):
        try:
            resultado = procesar_archivo_excel(archivo, tolerancia=tolerancia)
            resultados_ok.append(resultado)
        except Exception as e:
            errores.append({
                "archivo": archivo.name,
                "error": str(e)
            })
        progreso.progress(idx / total, text=f"Procesando archivos... {idx}/{total}")

    progreso.empty()

    c1, c2, c3 = st.columns(3)
    c1.metric("Archivos procesados bien", len(resultados_ok))
    c2.metric("Archivos con error", len(errores))
    c3.metric("Archivos cargados", len(archivos))

    if errores:
        st.warning("Algunos archivos no se pudieron procesar.")
        st.dataframe(pd.DataFrame(errores), use_container_width=True, hide_index=True)

    if resultados_ok:
        st.divider()
        st.subheader("Detalle por archivo")

        nombres = [r["archivo"] for r in resultados_ok]
        seleccionado = st.selectbox("Selecciona un archivo", nombres)

        r = next(x for x in resultados_ok if x["archivo"] == seleccionado)

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Problemáticas", len(r["analisis"]))
        cc2.metric("Participantes", len(r["participantes"]))
        cc3.metric("Instituciones únicas", len(r["instituciones"]))
        cc4.metric("Suma relaciones", int(r["matriz"].to_numpy().sum()))

        dd1, dd2, dd3 = st.columns(3)
        dd1.metric("Media influencia", f"{r['media_influencia']:.2f}")
        dd2.metric("Media dependencia", f"{r['media_dependencia']:.2f}")
        dd3.metric("Tolerancia", f"{r['tolerancia']:.2f}")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Participantes",
            "Instituciones",
            "Matriz limpia",
            "Análisis MIC MAC"
        ])

        with tab1:
            st.dataframe(r["participantes"], use_container_width=True, hide_index=True)

        with tab2:
            st.dataframe(r["instituciones"], use_container_width=True, hide_index=True)

        with tab3:
            st.dataframe(r["matriz"], use_container_width=True)

        with tab4:
            st.dataframe(r["analisis"], use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Ubicación por zona")

        for zona in ["PODER", "CONFLICTO", "RESULTADO", "AUTONOMA", "FRONTERA", "SIN RELACION"]:
            sub = r["analisis"][r["analisis"]["ZONA"] == zona].copy()
            if not sub.empty:
                st.markdown(f"### {zona}")
                st.dataframe(
                    sub[["CODIGO", "DESCRIPTOR", "CATEGORIA", "INFLUENCIA", "DEPENDENCIA"]],
                    use_container_width=True,
                    hide_index=True
                )

        st.divider()
        st.subheader("Visualizaciones")

        png_matriz = generar_png_matriz(r["matriz"])
        png_mapa = generar_png_mapa_micmac(
            r["analisis"],
            r["media_influencia"],
            r["media_dependencia"],
            r["tolerancia"]
        )

        vg1, vg2 = st.columns(2)
        with vg1:
            st.image(png_matriz, caption="Matriz MIC MAC", use_container_width=True)
        with vg2:
            st.image(png_mapa, caption="Mapa MIC MAC ajustado", use_container_width=True)

        st.divider()
        st.subheader("Descargas")

        excel_individual = exportar_excel_individual(
            nombre_archivo_fuente=r["archivo"],
            participantes_df=r["participantes"],
            instituciones_df=r["instituciones"],
            matriz_df=r["matriz"],
            df_micmac=r["analisis"],
            media_influencia=r["media_influencia"],
            media_dependencia=r["media_dependencia"],
            tolerancia=r["tolerancia"]
        )

        dg1, dg2, dg3 = st.columns(3)
        with dg1:
            st.download_button(
                "Descargar Excel individual",
                data=excel_individual,
                file_name=f"Resultados_{nombre_archivo_sin_extension(r['archivo'])}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        with dg2:
            st.download_button(
                "Descargar PNG matriz",
                data=png_matriz,
                file_name=f"Matriz_{nombre_archivo_sin_extension(r['archivo'])}.png",
                mime="image/png",
                use_container_width=True
            )
        with dg3:
            st.download_button(
                "Descargar PNG mapa",
                data=png_mapa,
                file_name=f"Mapa_{nombre_archivo_sin_extension(r['archivo'])}.png",
                mime="image/png",
                use_container_width=True
            )
