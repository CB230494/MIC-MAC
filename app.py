import io
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="MIC MAC independiente desde plantilla Excel",
    layout="wide"
)

COLOR_AZUL = "#1F77B4"
COLOR_VERDE = "#2CA02C"
COLOR_NARANJA = "#FF7F0E"
COLOR_ROJO = "#D62728"
COLOR_GRIS = "#7F7F7F"


def limpiar_texto(valor):
    if pd.isna(valor):
        return ""
    return str(valor).strip()


def es_vacio(valor):
    txt = limpiar_texto(valor)
    return txt == "" or txt.lower() == "nan"


def normalizar_codigo(valor):
    return limpiar_texto(valor).upper()


def detectar_hoja_descriptores(xls):
    for hoja in xls.sheet_names:
        if "DESCRIPTOR" in hoja.upper():
            return hoja
    return xls.sheet_names[0]


def leer_descriptores(xls):
    hoja = detectar_hoja_descriptores(xls)
    df = pd.read_excel(xls, sheet_name=hoja, header=0)

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
            "La hoja de descriptores no contiene las columnas esperadas 'NOMBRE CORTO' y 'DESCRIPTOR'."
        )

    base = df[[col_codigo, col_descriptor]].copy()

    if col_categoria:
        base["CATEGORIA"] = df[col_categoria]
    else:
        base["CATEGORIA"] = ""

    base.columns = ["CODIGO", "DESCRIPTOR", "CATEGORIA"]
    base["CODIGO"] = base["CODIGO"].apply(normalizar_codigo)
    base["DESCRIPTOR"] = base["DESCRIPTOR"].apply(limpiar_texto)
    base["CATEGORIA"] = base["CATEGORIA"].apply(limpiar_texto)

    base = base[
        (base["CODIGO"] != "") &
        (base["DESCRIPTOR"] != "")
    ].drop_duplicates(subset=["CODIGO"], keep="first").reset_index(drop=True)

    return base


def detectar_fila_participantes(df_raw):
    for i in range(len(df_raw)):
        fila = [limpiar_texto(x).lower() for x in df_raw.iloc[i].tolist()]
        texto_fila = " | ".join(fila)
        if "participante" in texto_fila and ("institución" in texto_fila or "institucion" in texto_fila):
            return i
    return None


def leer_participantes_instituciones(df_raw):
    fila_header = detectar_fila_participantes(df_raw)

    if fila_header is None:
        return pd.DataFrame(columns=["PARTICIPANTE", "INSTITUCION", "PUESTO"]), pd.DataFrame(columns=["INSTITUCION"])

    participantes = []
    i = fila_header + 1

    while i < len(df_raw):
        fila = df_raw.iloc[i]

        participante = limpiar_texto(fila.iloc[0]) if len(fila) > 0 else ""
        institucion = limpiar_texto(fila.iloc[4]) if len(fila) > 4 else ""
        puesto = limpiar_texto(fila.iloc[8]) if len(fila) > 8 else ""

        fila_completa_vacia = all(es_vacio(v) for v in fila.tolist())

        if fila_completa_vacia:
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


def detectar_hoja_matriz(xls):
    for hoja in xls.sheet_names:
        if "MATRIZ" in hoja.upper():
            return hoja
    return None


def detectar_fila_encabezado_matriz(df_raw):
    for i in range(len(df_raw)):
        fila = [limpiar_texto(x) for x in df_raw.iloc[i].tolist()]

        if len(fila) < 4:
            continue

        primera = fila[0]
        resto = fila[1:]
        no_vacios = [normalizar_codigo(x) for x in resto if not es_vacio(x)]

        if es_vacio(primera) and len(no_vacios) >= 3:
            if len(set(no_vacios)) >= 3:
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

        codigo_fila = normalizar_codigo(fila[0])
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


def clasificar_zona(inf, dep, media_inf, media_dep):
    if inf == 0 and dep == 0:
        return "SIN RELACION"
    if inf > media_inf and dep < media_dep:
        return "PODER"
    if inf > media_inf and dep > media_dep:
        return "CONFLICTO"
    if inf < media_inf and dep > media_dep:
        return "RESULTADO"
    return "AUTONOMA"


def calcular_micmac(matriz_df, analisis_base):
    influencia = matriz_df.sum(axis=1)
    dependencia = matriz_df.sum(axis=0)

    df = analisis_base.copy()
    df["INFLUENCIA"] = df["CODIGO"].map(influencia.to_dict()).fillna(0).astype(int)
    df["DEPENDENCIA"] = df["CODIGO"].map(dependencia.to_dict()).fillna(0).astype(int)

    media_influencia = float(df["INFLUENCIA"].mean()) if not df.empty else 0.0
    media_dependencia = float(df["DEPENDENCIA"].mean()) if not df.empty else 0.0

    df["ZONA"] = df.apply(
        lambda row: clasificar_zona(
            row["INFLUENCIA"],
            row["DEPENDENCIA"],
            media_influencia,
            media_dependencia
        ),
        axis=1
    )

    orden_zona = {
        "PODER": 1,
        "CONFLICTO": 2,
        "RESULTADO": 3,
        "AUTONOMA": 4,
        "SIN RELACION": 5
    }

    df["ORDEN_ZONA"] = df["ZONA"].map(orden_zona).fillna(99)
    df = df.sort_values(
        by=["ORDEN_ZONA", "INFLUENCIA", "DEPENDENCIA", "DESCRIPTOR"],
        ascending=[True, False, False, True]
    ).reset_index(drop=True)
    df = df.drop(columns=["ORDEN_ZONA"])

    return df, media_influencia, media_dependencia


def construir_listado_zonas(df_micmac):
    orden = ["PODER", "CONFLICTO", "RESULTADO", "AUTONOMA", "SIN RELACION"]
    bloques = []

    for zona in orden:
        sub = df_micmac[df_micmac["ZONA"] == zona].copy()
        if not sub.empty:
            bloques.append((zona, sub))

    return bloques


def color_zona(zona):
    if zona == "PODER":
        return COLOR_VERDE
    if zona == "CONFLICTO":
        return COLOR_ROJO
    if zona == "RESULTADO":
        return COLOR_AZUL
    if zona == "AUTONOMA":
        return COLOR_NARANJA
    return COLOR_GRIS


def generar_png_mapa_micmac(df_micmac, media_influencia, media_dependencia):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

    for _, row in df_micmac.iterrows():
        x = row["DEPENDENCIA"]
        y = row["INFLUENCIA"]
        z = row["ZONA"]

        ax.scatter(
            x, y,
            s=180,
            color=color_zona(z),
            edgecolors="black",
            linewidths=0.7,
            alpha=0.9
        )

        ax.text(
            x + 0.12,
            y + 0.12,
            row["CODIGO"],
            fontsize=8,
            ha="left",
            va="bottom"
        )

    ax.axvline(media_dependencia, linestyle="--", color="gray", linewidth=1)
    ax.axhline(media_influencia, linestyle="--", color="gray", linewidth=1)

    ax.set_title("Mapa MIC MAC", fontsize=18)
    ax.set_xlabel("Dependencia")
    ax.set_ylabel("Influencia")
    ax.grid(True, alpha=0.25)

    max_x = max(float(df_micmac["DEPENDENCIA"].max()), float(media_dependencia)) + 2 if not df_micmac.empty else 5
    max_y = max(float(df_micmac["INFLUENCIA"].max()), float(media_influencia)) + 2 if not df_micmac.empty else 5
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def generar_png_matriz(matriz_df):
    fig_w = max(8, len(matriz_df.columns) * 0.45)
    fig_h = max(6, len(matriz_df.index) * 0.42)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

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
                fontsize=7
            )

    ax.set_title("Matriz MIC MAC", fontsize=16)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def exportar_excel_resultados(
    nombre_archivo_fuente,
    participantes_df,
    instituciones_df,
    matriz_df,
    df_micmac,
    media_influencia,
    media_dependencia
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
                "Método de corte"
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
                "Media aritmética"
            ]
        })
        resumen.to_excel(writer, sheet_name="RESUMEN", index=False)

        participantes_df.to_excel(writer, sheet_name="PARTICIPANTES", index=False)
        instituciones_df.to_excel(writer, sheet_name="INSTITUCIONES", index=False)

        matriz_export = matriz_df.copy()
        matriz_export.insert(0, "CODIGO", matriz_export.index)
        matriz_export.to_excel(writer, sheet_name="MATRIZ_LIMPIA", index=False)

        df_micmac.to_excel(writer, sheet_name="ANALISIS_MICMAC", index=False)

        zonas = ["PODER", "CONFLICTO", "RESULTADO", "AUTONOMA", "SIN RELACION"]
        filas_zona = []

        for zona in zonas:
            sub = df_micmac[df_micmac["ZONA"] == zona].copy()
            if not sub.empty:
                for _, row in sub.iterrows():
                    filas_zona.append({
                        "ZONA": zona,
                        "CODIGO": row["CODIGO"],
                        "DESCRIPTOR": row["DESCRIPTOR"],
                        "CATEGORIA": row["CATEGORIA"],
                        "INFLUENCIA": row["INFLUENCIA"],
                        "DEPENDENCIA": row["DEPENDENCIA"]
                    })

        pd.DataFrame(filas_zona).to_excel(writer, sheet_name="PROBLEMATICAS_POR_ZONA", index=False)

        hojas = {
            "RESUMEN": resumen,
            "PARTICIPANTES": participantes_df,
            "INSTITUCIONES": instituciones_df,
            "MATRIZ_LIMPIA": matriz_export,
            "ANALISIS_MICMAC": df_micmac,
            "PROBLEMATICAS_POR_ZONA": pd.DataFrame(filas_zona)
        }

        for hoja, df in hojas.items():
            ws = writer.sheets[hoja]
            if df.empty:
                continue
            for idx, col in enumerate(df.columns):
                max_len = max(
                    len(str(col)),
                    df[col].astype(str).map(len).max() if len(df) > 0 else 0
                )
                ws.set_column(idx, idx, min(max(max_len + 2, 14), 55))

    return salida.getvalue()


def procesar_archivo_excel(file):
    xls = pd.ExcelFile(file)

    hoja_matriz = detectar_hoja_matriz(xls)
    if hoja_matriz is None:
        raise ValueError("No se encontró una hoja llamada 'MATRIZ' en el archivo.")

    df_raw_matriz = pd.read_excel(xls, sheet_name=hoja_matriz, header=None)
    catalogo = leer_descriptores(xls)
    participantes_df, instituciones_df = leer_participantes_instituciones(df_raw_matriz)
    matriz_df, analisis_base = leer_matriz_micmac(df_raw_matriz, catalogo)
    df_micmac, media_influencia, media_dependencia = calcular_micmac(matriz_df, analisis_base)

    return {
        "participantes": participantes_df,
        "instituciones": instituciones_df,
        "matriz": matriz_df,
        "analisis": df_micmac,
        "media_influencia": media_influencia,
        "media_dependencia": media_dependencia
    }


st.title("MIC MAC independiente desde plantilla Excel")
st.caption("Sube la plantilla MIC MAC. La app leerá cruces, depurará instituciones duplicadas y generará el análisis por zona.")

archivo = st.file_uploader(
    "Carga la plantilla MIC MAC en Excel",
    type=["xlsx"]
)

if archivo is not None:
    try:
        resultado = procesar_archivo_excel(archivo)

        participantes_df = resultado["participantes"]
        instituciones_df = resultado["instituciones"]
        matriz_df = resultado["matriz"]
        df_micmac = resultado["analisis"]
        media_influencia = resultado["media_influencia"]
        media_dependencia = resultado["media_dependencia"]

        st.success("Plantilla procesada correctamente.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Problemáticas", len(df_micmac))
        c2.metric("Participantes", len(participantes_df))
        c3.metric("Instituciones únicas", len(instituciones_df))
        c4.metric("Suma relaciones", int(matriz_df.to_numpy().sum()))

        st.divider()

        st.markdown("### Líneas de corte utilizadas")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Media influencia", f"{media_influencia:.2f}")
        cc2.metric("Media dependencia", f"{media_dependencia:.2f}")
        cc3.metric("Método", "Media aritmética")

        st.divider()

        t1, t2, t3, t4 = st.tabs([
            "Participantes",
            "Instituciones",
            "Matriz limpia",
            "Análisis MIC MAC"
        ])

        with t1:
            st.subheader("Participantes detectados")
            st.dataframe(participantes_df, use_container_width=True, hide_index=True)

        with t2:
            st.subheader("Instituciones únicas")
            st.dataframe(instituciones_df, use_container_width=True, hide_index=True)

        with t3:
            st.subheader("Matriz MIC MAC limpia")
            st.dataframe(matriz_df, use_container_width=True)

        with t4:
            st.subheader("Resultado MIC MAC")
            st.dataframe(df_micmac, use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("Ubicación de las problemáticas por zona")
        bloques = construir_listado_zonas(df_micmac)

        for zona, sub in bloques:
            st.markdown(f"### {zona}")
            st.dataframe(
                sub[["CODIGO", "DESCRIPTOR", "CATEGORIA", "INFLUENCIA", "DEPENDENCIA"]],
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        st.subheader("Visualizaciones")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            png_matriz = generar_png_matriz(matriz_df)
            st.image(png_matriz, caption="Matriz MIC MAC", use_container_width=True)

        with col_g2:
            png_mapa = generar_png_mapa_micmac(df_micmac, media_influencia, media_dependencia)
            st.image(png_mapa, caption="Mapa MIC MAC", use_container_width=True)

        st.divider()

        st.subheader("Descargas")

        excel_resultado = exportar_excel_resultados(
            nombre_archivo_fuente=archivo.name,
            participantes_df=participantes_df,
            instituciones_df=instituciones_df,
            matriz_df=matriz_df,
            df_micmac=df_micmac,
            media_influencia=media_influencia,
            media_dependencia=media_dependencia
        )

        d1, d2, d3 = st.columns(3)

        with d1:
            st.download_button(
                "Descargar Excel de resultados",
                data=excel_resultado,
                file_name=f"Resultados_MICMAC_{archivo.name.replace('.xlsx', '')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with d2:
            st.download_button(
                "Descargar PNG matriz",
                data=png_matriz,
                file_name="Matriz_MICMAC.png",
                mime="image/png",
                use_container_width=True
            )

        with d3:
            st.download_button(
                "Descargar PNG mapa",
                data=png_mapa,
                file_name="Mapa_MICMAC.png",
                mime="image/png",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"No se pudo procesar la plantilla: {e}")
