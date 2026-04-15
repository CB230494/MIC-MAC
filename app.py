import io
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Procesador MIC MAC",
    layout="wide"
)

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
# INFLUENCIAS POR COLUMNA
# =========================================================
def construir_influencias_por_columna(matriz_df, analisis_base):
    if matriz_df is None or matriz_df.empty:
        vacio_det = pd.DataFrame(columns=[
            "COLUMNA_CODIGO", "COLUMNA_DESCRIPTOR", "COLUMNA_CATEGORIA",
            "FILA_CODIGO", "FILA_DESCRIPTOR", "FILA_CATEGORIA", "VALOR"
        ])
        vacio_res = pd.DataFrame(columns=[
            "COLUMNA_CODIGO", "COLUMNA_DESCRIPTOR", "COLUMNA_CATEGORIA",
            "INFLUYEN_3", "INFLUYEN_2"
        ])
        return vacio_det, vacio_res

    mapa_desc = dict(zip(analisis_base["CODIGO"], analisis_base["DESCRIPTOR"]))
    mapa_cat = dict(zip(analisis_base["CODIGO"], analisis_base["CATEGORIA"]))

    detalle = []
    resumen = []

    for col_codigo in matriz_df.columns:
        serie_col = matriz_df[col_codigo]

        filas_3_cod = []
        filas_3_desc = []
        filas_2_cod = []
        filas_2_desc = []

        for fila_codigo, valor in serie_col.items():
            val = int(pd.to_numeric(valor, errors="coerce") or 0)

            if val == 3:
                filas_3_cod.append(fila_codigo)
                filas_3_desc.append(mapa_desc.get(fila_codigo, fila_codigo))
                detalle.append({
                    "COLUMNA_CODIGO": col_codigo,
                    "COLUMNA_DESCRIPTOR": mapa_desc.get(col_codigo, col_codigo),
                    "COLUMNA_CATEGORIA": mapa_cat.get(col_codigo, ""),
                    "FILA_CODIGO": fila_codigo,
                    "FILA_DESCRIPTOR": mapa_desc.get(fila_codigo, fila_codigo),
                    "FILA_CATEGORIA": mapa_cat.get(fila_codigo, ""),
                    "VALOR": 3
                })

            elif val == 2:
                filas_2_cod.append(fila_codigo)
                filas_2_desc.append(mapa_desc.get(fila_codigo, fila_codigo))
                detalle.append({
                    "COLUMNA_CODIGO": col_codigo,
                    "COLUMNA_DESCRIPTOR": mapa_desc.get(col_codigo, col_codigo),
                    "COLUMNA_CATEGORIA": mapa_cat.get(col_codigo, ""),
                    "FILA_CODIGO": fila_codigo,
                    "FILA_DESCRIPTOR": mapa_desc.get(fila_codigo, fila_codigo),
                    "FILA_CATEGORIA": mapa_cat.get(fila_codigo, ""),
                    "VALOR": 2
                })

        resumen.append({
            "COLUMNA_CODIGO": col_codigo,
            "COLUMNA_DESCRIPTOR": mapa_desc.get(col_codigo, col_codigo),
            "COLUMNA_CATEGORIA": mapa_cat.get(col_codigo, ""),
            "INFLUYEN_3": " | ".join(filas_3_cod) if filas_3_cod else "",
            "DESCRIPTORES_3": " | ".join(filas_3_desc) if filas_3_desc else "",
            "INFLUYEN_2": " | ".join(filas_2_cod) if filas_2_cod else "",
            "DESCRIPTORES_2": " | ".join(filas_2_desc) if filas_2_desc else ""
        })

    detalle_df = pd.DataFrame(detalle)
    resumen_df = pd.DataFrame(resumen)

    if not detalle_df.empty:
        detalle_df = detalle_df.sort_values(
            by=["COLUMNA_DESCRIPTOR", "VALOR", "FILA_DESCRIPTOR"],
            ascending=[True, False, True]
        ).reset_index(drop=True)

    if not resumen_df.empty:
        resumen_df = resumen_df.sort_values(
            by=["COLUMNA_DESCRIPTOR"],
            ascending=[True]
        ).reset_index(drop=True)

    return detalle_df, resumen_df


# =========================================================
# EXPORTACIÓN
# =========================================================
def exportar_excel_individual(
    nombre_archivo_fuente,
    participantes_df,
    instituciones_df,
    influencias_detalle_df,
    influencias_resumen_df
):
    salida = io.BytesIO()

    with pd.ExcelWriter(salida, engine="xlsxwriter") as writer:
        resumen = pd.DataFrame({
            "CAMPO": [
                "Archivo fuente",
                "Fecha de procesamiento",
                "Cantidad de participantes",
                "Cantidad de instituciones únicas",
                "Cantidad relaciones valor 3",
                "Cantidad relaciones valor 2"
            ],
            "VALOR": [
                nombre_archivo_fuente,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(participantes_df),
                len(instituciones_df),
                int((influencias_detalle_df["VALOR"] == 3).sum()) if not influencias_detalle_df.empty else 0,
                int((influencias_detalle_df["VALOR"] == 2).sum()) if not influencias_detalle_df.empty else 0
            ]
        })

        resumen.to_excel(writer, sheet_name="RESUMEN", index=False)
        participantes_df.to_excel(writer, sheet_name="ACTORES", index=False)
        instituciones_df.to_excel(writer, sheet_name="INSTITUCIONES", index=False)
        influencias_resumen_df.to_excel(writer, sheet_name="INFLUYENCIAS_RESUMEN", index=False)
        influencias_detalle_df.to_excel(writer, sheet_name="INFLUYENCIAS_DETALLE", index=False)

        hojas = {
            "RESUMEN": resumen,
            "ACTORES": participantes_df,
            "INSTITUCIONES": instituciones_df,
            "INFLUYENCIAS_RESUMEN": influencias_resumen_df,
            "INFLUYENCIAS_DETALLE": influencias_detalle_df
        }

        for hoja, df in hojas.items():
            ws = writer.sheets[hoja]
            if df.empty:
                continue
            for idx, col in enumerate(df.columns):
                max_len = max(len(str(col)), df[col].astype(str).map(len).max() if len(df) > 0 else 0)
                ws.set_column(idx, idx, min(max(max_len + 2, 14), 60))

    return salida.getvalue()


# =========================================================
# PROCESAMIENTO CENTRAL
# =========================================================
def procesar_archivo_excel(file):
    xls = pd.ExcelFile(file)

    hoja_matriz = detectar_hoja_matriz(xls)
    if hoja_matriz is None:
        raise ValueError("No se encontró una hoja de matriz válida.")

    catalogo, hoja_desc = leer_descriptores(xls)
    df_raw_matriz = pd.read_excel(xls, sheet_name=hoja_matriz, header=None)

    participantes_df, instituciones_df = leer_participantes_instituciones(df_raw_matriz)
    matriz_df, analisis_base = leer_matriz_micmac(df_raw_matriz, catalogo)

    influencias_detalle_df, influencias_resumen_df = construir_influencias_por_columna(
        matriz_df,
        analisis_base
    )

    return {
        "archivo": file.name,
        "hoja_descriptores": hoja_desc,
        "hoja_matriz": hoja_matriz,
        "catalogo": catalogo,
        "participantes": participantes_df,
        "instituciones": instituciones_df,
        "matriz": matriz_df,
        "influencias_detalle": influencias_detalle_df,
        "influencias_resumen": influencias_resumen_df
    }


# =========================================================
# INTERFAZ
# =========================================================
st.title("Procesador MIC MAC")
st.caption(
    "Sube uno o varios Excel. Esta versión muestra únicamente actores e influencias con valor 3 y 2 sobre cada columna."
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
            resultado = procesar_archivo_excel(archivo)
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
        cc1.metric("Participantes", len(r["participantes"]))
        cc2.metric("Instituciones únicas", len(r["instituciones"]))
        cc3.metric(
            "Relaciones valor 3",
            int((r["influencias_detalle"]["VALOR"] == 3).sum()) if not r["influencias_detalle"].empty else 0
        )
        cc4.metric(
            "Relaciones valor 2",
            int((r["influencias_detalle"]["VALOR"] == 2).sum()) if not r["influencias_detalle"].empty else 0
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "Actores",
            "Instituciones",
            "Influyen con 3 y 2",
            "Detalle completo"
        ])

        with tab1:
            st.dataframe(r["participantes"], use_container_width=True, hide_index=True)

        with tab2:
            st.dataframe(r["instituciones"], use_container_width=True, hide_index=True)

        with tab3:
            st.dataframe(r["influencias_resumen"], use_container_width=True, hide_index=True)

        with tab4:
            st.dataframe(r["influencias_detalle"], use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Descargas")

        excel_individual = exportar_excel_individual(
            nombre_archivo_fuente=r["archivo"],
            participantes_df=r["participantes"],
            instituciones_df=r["instituciones"],
            influencias_detalle_df=r["influencias_detalle"],
            influencias_resumen_df=r["influencias_resumen"]
        )

        st.download_button(
            "Descargar Excel",
            data=excel_individual,
            file_name=f"Resultados_{nombre_archivo_sin_extension(r['archivo'])}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
else:
    st.info("Carga uno o varios archivos Excel para comenzar.")
