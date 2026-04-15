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
# LECTURA: FILA -> COLUMNA
# =========================================================
def construir_influencias_por_columna(matriz_df, analisis_base):
    if matriz_df is None or matriz_df.empty:
        vacio_det = pd.DataFrame(columns=[
            "COLUMNA_CODIGO", "COLUMNA_DESCRIPTOR", "COLUMNA_CATEGORIA",
            "FILA_CODIGO", "FILA_DESCRIPTOR", "FILA_CATEGORIA", "VALOR"
        ])
        vacio_res = pd.DataFrame(columns=[
            "COLUMNA_CODIGO", "COLUMNA_DESCRIPTOR", "COLUMNA_CATEGORIA",
            "INFLUYEN_3", "DESCRIPTORES_3", "INFLUYEN_2", "DESCRIPTORES_2"
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
        wb = writer.book

        fmt_titulo = wb.add_format({
            "bold": True,
            "font_size": 14,
            "font_color": "white",
            "bg_color": "#0B3954",
            "align": "center",
            "valign": "vcenter",
            "border": 1
        })

        fmt_head = wb.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#1F5F8B",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "text_wrap": True
        })

        fmt_head_verde = wb.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#1B9E77",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "text_wrap": True
        })

        fmt_head_naranja = wb.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#E67E22",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "text_wrap": True
        })

        fmt_texto = wb.add_format({
            "border": 1,
            "valign": "top",
            "text_wrap": True
        })

        fmt_texto_centro = wb.add_format({
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True
        })

        fmt_texto_gris = wb.add_format({
            "border": 1,
            "valign": "top",
            "text_wrap": True,
            "bg_color": "#F3F4F6"
        })

        fmt_valor_3 = wb.add_format({
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "bold": True,
            "font_color": "white",
            "bg_color": "#1B9E77"
        })

        fmt_valor_2 = wb.add_format({
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "bold": True,
            "font_color": "white",
            "bg_color": "#E67E22"
        })

        fmt_resumen_label = wb.add_format({
            "bold": True,
            "bg_color": "#EAF3FB",
            "border": 1,
            "text_wrap": True
        })

        fmt_resumen_value = wb.add_format({
            "border": 1,
            "text_wrap": True
        })

        participantes_df = participantes_df.copy()
        instituciones_df = instituciones_df.copy()
        influencias_detalle_df = influencias_detalle_df.copy()
        influencias_resumen_df = influencias_resumen_df.copy()

        if not influencias_detalle_df.empty:
            influencias_detalle_df = influencias_detalle_df.sort_values(
                by=["COLUMNA_DESCRIPTOR", "VALOR", "FILA_DESCRIPTOR"],
                ascending=[True, False, True]
            ).reset_index(drop=True)

        if not influencias_resumen_df.empty:
            influencias_resumen_df = influencias_resumen_df.sort_values(
                by=["COLUMNA_DESCRIPTOR"],
                ascending=[True]
            ).reset_index(drop=True)

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

        # RESUMEN
        ws_res = wb.add_worksheet("RESUMEN")
        writer.sheets["RESUMEN"] = ws_res

        ws_res.merge_range("A1:B1", "RESUMEN DEL PROCESAMIENTO", fmt_titulo)
        ws_res.write("A3", "CAMPO", fmt_head)
        ws_res.write("B3", "VALOR", fmt_head)

        for i, (_, row) in enumerate(resumen.iterrows(), start=3):
            ws_res.write(i, 0, row["CAMPO"], fmt_resumen_label)
            ws_res.write(i, 1, row["VALOR"], fmt_resumen_value)

        ws_res.set_column("A:A", 35)
        ws_res.set_column("B:B", 35)
        ws_res.freeze_panes(3, 0)

        # ACTORES
        ws_act = wb.add_worksheet("ACTORES")
        writer.sheets["ACTORES"] = ws_act

        ws_act.merge_range("A1:C1", "LISTA DE ACTORES / PARTICIPANTES", fmt_titulo)

        headers_act = ["PARTICIPANTE", "INSTITUCIÓN", "PUESTO / CARGO"]
        for col, h in enumerate(headers_act):
            ws_act.write(2, col, h, fmt_head)

        if participantes_df.empty:
            ws_act.write(3, 0, "Sin datos", fmt_texto_gris)
        else:
            for r_idx, (_, row) in enumerate(participantes_df.iterrows(), start=3):
                ws_act.write(r_idx, 0, row.get("PARTICIPANTE", ""), fmt_texto)
                ws_act.write(r_idx, 1, row.get("INSTITUCION", ""), fmt_texto)
                ws_act.write(r_idx, 2, row.get("PUESTO", ""), fmt_texto)

            last_row = len(participantes_df) + 2
            ws_act.autofilter(2, 0, last_row, 2)

        ws_act.set_column("A:A", 32)
        ws_act.set_column("B:B", 34)
        ws_act.set_column("C:C", 30)
        ws_act.freeze_panes(3, 0)

        # INSTITUCIONES
        ws_inst = wb.add_worksheet("INSTITUCIONES")
        writer.sheets["INSTITUCIONES"] = ws_inst

        ws_inst.write("A1", "INSTITUCIONES IDENTIFICADAS", fmt_titulo)
        ws_inst.write("A3", "INSTITUCIÓN", fmt_head)

        if instituciones_df.empty:
            ws_inst.write(3, 0, "Sin datos", fmt_texto_gris)
        else:
            for r_idx, (_, row) in enumerate(instituciones_df.iterrows(), start=3):
                ws_inst.write(r_idx, 0, row.get("INSTITUCION", ""), fmt_texto)

            last_row = len(instituciones_df) + 2
            ws_inst.autofilter(2, 0, last_row, 0)

        ws_inst.set_column("A:A", 40)
        ws_inst.freeze_panes(3, 0)

        # INFLUYENCIAS RESUMEN
        ws_ir = wb.add_worksheet("INFLUYENCIAS_RESUMEN")
        writer.sheets["INFLUYENCIAS_RESUMEN"] = ws_ir

        ws_ir.merge_range("A1:G1", "RESUMEN DE INFLUENCIAS FILA → COLUMNA", fmt_titulo)

        headers_ir = [
            "CÓDIGO COLUMNA",
            "DESCRIPTOR COLUMNA",
            "CATEGORÍA",
            "INFLUYEN CON 3",
            "DESCRIPTORES CON 3",
            "INFLUYEN CON 2",
            "DESCRIPTORES CON 2"
        ]

        formatos_headers_ir = [
            fmt_head, fmt_head, fmt_head,
            fmt_head_verde, fmt_head_verde,
            fmt_head_naranja, fmt_head_naranja
        ]

        for col, h in enumerate(headers_ir):
            ws_ir.write(2, col, h, formatos_headers_ir[col])

        if influencias_resumen_df.empty:
            ws_ir.write(3, 0, "Sin datos", fmt_texto_gris)
        else:
            for r_idx, (_, row) in enumerate(influencias_resumen_df.iterrows(), start=3):
                ws_ir.write(r_idx, 0, row.get("COLUMNA_CODIGO", ""), fmt_texto_centro)
                ws_ir.write(r_idx, 1, row.get("COLUMNA_DESCRIPTOR", ""), fmt_texto)
                ws_ir.write(r_idx, 2, row.get("COLUMNA_CATEGORIA", ""), fmt_texto)
                ws_ir.write(r_idx, 3, row.get("INFLUYEN_3", ""), fmt_texto)
                ws_ir.write(r_idx, 4, row.get("DESCRIPTORES_3", ""), fmt_texto)
                ws_ir.write(r_idx, 5, row.get("INFLUYEN_2", ""), fmt_texto)
                ws_ir.write(r_idx, 6, row.get("DESCRIPTORES_2", ""), fmt_texto)

            last_row = len(influencias_resumen_df) + 2
            ws_ir.autofilter(2, 0, last_row, 6)

        ws_ir.set_column("A:A", 18)
        ws_ir.set_column("B:B", 32)
        ws_ir.set_column("C:C", 24)
        ws_ir.set_column("D:D", 28)
        ws_ir.set_column("E:E", 40)
        ws_ir.set_column("F:F", 28)
        ws_ir.set_column("G:G", 40)
        ws_ir.freeze_panes(3, 0)

        # INFLUYENCIAS DETALLE
        ws_id = wb.add_worksheet("INFLUYENCIAS_DETALLE")
        writer.sheets["INFLUYENCIAS_DETALLE"] = ws_id

        ws_id.merge_range("A1:G1", "DETALLE DE INFLUENCIAS FILA → COLUMNA", fmt_titulo)

        headers_id = [
            "CÓDIGO COLUMNA",
            "DESCRIPTOR COLUMNA",
            "CATEGORÍA COLUMNA",
            "CÓDIGO FILA",
            "DESCRIPTOR FILA",
            "CATEGORÍA FILA",
            "VALOR"
        ]

        for col, h in enumerate(headers_id):
            ws_id.write(2, col, h, fmt_head)

        if influencias_detalle_df.empty:
            ws_id.write(3, 0, "Sin datos", fmt_texto_gris)
        else:
            for r_idx, (_, row) in enumerate(influencias_detalle_df.iterrows(), start=3):
                ws_id.write(r_idx, 0, row.get("COLUMNA_CODIGO", ""), fmt_texto_centro)
                ws_id.write(r_idx, 1, row.get("COLUMNA_DESCRIPTOR", ""), fmt_texto)
                ws_id.write(r_idx, 2, row.get("COLUMNA_CATEGORIA", ""), fmt_texto)
                ws_id.write(r_idx, 3, row.get("FILA_CODIGO", ""), fmt_texto_centro)
                ws_id.write(r_idx, 4, row.get("FILA_DESCRIPTOR", ""), fmt_texto)
                ws_id.write(r_idx, 5, row.get("FILA_CATEGORIA", ""), fmt_texto)

                valor = row.get("VALOR", "")
                if valor == 3:
                    ws_id.write(r_idx, 6, valor, fmt_valor_3)
                elif valor == 2:
                    ws_id.write(r_idx, 6, valor, fmt_valor_2)
                else:
                    ws_id.write(r_idx, 6, valor, fmt_texto_centro)

            last_row = len(influencias_detalle_df) + 2
            ws_id.autofilter(2, 0, last_row, 6)

        ws_id.set_column("A:A", 18)
        ws_id.set_column("B:B", 30)
        ws_id.set_column("C:C", 24)
        ws_id.set_column("D:D", 16)
        ws_id.set_column("E:E", 30)
        ws_id.set_column("F:F", 24)
        ws_id.set_column("G:G", 10)
        ws_id.freeze_panes(3, 0)

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
        "participantes": participantes_df,
        "instituciones": instituciones_df,
        "influencias_detalle": influencias_detalle_df,
        "influencias_resumen": influencias_resumen_df
    }


# =========================================================
# INTERFAZ
# =========================================================
st.title("Procesador MIC MAC")
st.caption(
    "Carga uno o varios Excel. Esta versión muestra únicamente actores, instituciones e influencias de fila sobre columna con valores 3 y 2."
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
