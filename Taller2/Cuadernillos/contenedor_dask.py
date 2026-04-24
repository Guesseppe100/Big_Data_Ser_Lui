# ============================================================
# contenedor_dask.py
# CONTENEDOR SECOP II - DASK
# ============================================================

import re
import time
import unicodedata
import threading
import concurrent.futures
from io import StringIO
from pathlib import Path

import requests
import pandas as pd
import dask.dataframe as dd


class CONTENEDOR_SECOP:
    """
    Contenedor funcional para descarga, carga, limpieza y preparación
    del dataset SECOP II usando Dask DataFrame.
    """

    # ========================================================
    # 1. CONSTRUCTOR
    # ========================================================

    def __init__(self):

        self.chunk_default = 100_000
        self.total_filas_default = 8_410_000

        self.columnas_q1 = [
            "departamento",
            "valor_adjudicado",
            "fecha_publicacion",
            "estado_adjudicacion"
        ]

        self.diccionario_semantico = {
            "departamento_entidad": "departamento",
            "valor_total_adjudicacion": "valor_adjudicado",
            "fecha_de_publicacion_del": "fecha_publicacion",
            "adjudicado": "estado_adjudicacion",

            "entidad": "nombre_entidad",
            "nit_entidad": "nit_entidad",
            "ciudad_entidad": "ciudad_entidad",
            "modalidad_de_contratacion": "modalidad_contratacion",
            "estado_del_procedimiento": "estado_procedimiento",
            "tipo_de_contrato": "tipo_contrato",
            "subtipo_de_contrato": "subtipo_contrato",
            "departamento_proveedor": "departamento_proveedor",
            "ciudad_proveedor": "ciudad_proveedor",
            "nombre_del_proveedor": "nombre_proveedor",
            "nit_del_proveedor_adjudicado": "nit_proveedor",
            "fecha_adjudicacion": "fecha_adjudicacion",
            "precio_base": "precio_base",
            "codigo_principal_de_categoria": "codigo_categoria_unspsc",
            "urlproceso": "url_proceso"
        }

    # ========================================================
    # 2. DESCARGA INDIVIDUAL POR CHUNK
    # ========================================================

    def descargar_chunk_secop(
        self,
        offset,
        base_url,
        data_dir,
        chunk=None,
        timeout=300
    ):

        if chunk is None:
            chunk = self.chunk_default

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        ruta = data_dir / f"secop_chunk_{offset:07d}.csv"

        if ruta.exists() and ruta.stat().st_size > 0:
            print(f"[SKIP] Ya existe: {ruta.name}")

            return {
                "offset": offset,
                "filas": None,
                "columnas": None,
                "mem_mb": None,
                "ruta": ruta,
                "estado": "existente"
            }

        url = f"{base_url}?$limit={chunk}&$offset={offset}"

        hilo_nombre = threading.current_thread().name
        hilo_num = hilo_nombre.split("_")[-1] if "_" in hilo_nombre else "0"

        print(f"[Hilo-{hilo_num}] Iniciando offset={offset:,}")

        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()

        df = pd.read_csv(StringIO(resp.text), low_memory=False)

        df.to_csv(ruta, index=False)

        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        tam_mb = ruta.stat().st_size / (1024 * 1024)

        print(
            f"[Hilo-{hilo_num}] OK offset={offset:,}: "
            f"{len(df):,} filas | {mem_mb:.1f} MB RAM | "
            f"{tam_mb:.1f} MB disco -> {ruta.name}"
        )

        return {
            "offset": offset,
            "filas": len(df),
            "columnas": df.shape[1],
            "mem_mb": mem_mb,
            "ruta": ruta,
            "estado": "descargado"
        }

    # ========================================================
    # 3. DESCARGA COMPLETA DEL DATASET
    # ========================================================

    def descargar_dataset_secop(
        self,
        base_url,
        data_dir,
        total_filas=None,
        chunk=None,
        max_workers=2,
        timeout=300
    ):

        if total_filas is None:
            total_filas = self.total_filas_default

        if chunk is None:
            chunk = self.chunk_default

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        offsets = list(range(0, total_filas, chunk))

        print(f"Descargando {total_filas:,} filas en chunks de {chunk:,}")
        print(f"Carpeta destino: {data_dir}")
        print(f"Offsets generados: {len(offsets)}")
        print(f"Workers: {max_workers}")
        print()

        t0 = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:

            resultados = list(
                executor.map(
                    lambda offset: self.descargar_chunk_secop(
                        offset=offset,
                        base_url=base_url,
                        data_dir=data_dir,
                        chunk=chunk,
                        timeout=timeout
                    ),
                    offsets
                )
            )

        t_total = time.time() - t0

        resumen = pd.DataFrame(resultados)

        csvs = sorted(data_dir.glob("secop_chunk_*.csv"))

        print()
        print("=" * 75)
        print("RESUMEN DE DESCARGA")
        print("=" * 75)

        print("\nEstados de ejecución:")
        print(
            resumen["estado"]
            .value_counts(dropna=False)
        )

        descargados = (resumen["estado"] == "descargado").sum()
        existentes = (resumen["estado"] == "existente").sum()

        print("\nMétricas:")
        print(f"Chunks descargados nueva ejecución : {descargados}")
        print(f"Chunks previamente existentes      : {existentes}")
        print(f"Total archivos detectados          : {len(csvs)}")

        if "filas" in resumen.columns:

            filas_descargadas = (
                resumen["filas"]
                .dropna()
                .sum()
            )

            print(
                f"Filas descargadas esta ejecución   : "
                f"{filas_descargadas:,.0f}"
            )

        particiones_esperadas = len(offsets)

        print()
        print(f"Particiones esperadas              : {particiones_esperadas}")
        print(f"Particiones reales                 : {len(csvs)}")

        if len(csvs) != particiones_esperadas:
            print("ADVERTENCIA: el número de archivos no coincide con lo esperado.")

        tam_total_gb = sum(
            f.stat().st_size for f in csvs
        ) / (1024 ** 3)

        print(
            f"Tamaño total en disco              : "
            f"{tam_total_gb:.2f} GB"
        )

        print("\nPrimeros archivos detectados:")

        for f in csvs[:10]:
            tam_mb = f.stat().st_size / (1024 ** 2)
            print(f"{f.name} ({tam_mb:.1f} MB)")

        if len(csvs) > 10:
            print(f"... {len(csvs) - 10} archivos adicionales")

        print("\nTiempo total ejecución:")
        print(f"{t_total:.1f} segundos")

        print(
            f"\n-> Dask leerá {len(csvs)} archivos como particiones."
        )

        return resumen

    # ========================================================
    # 4. CARGA CON DASK
    # ========================================================

    def cargar_dask_dataframe(
        self,
        data_dir,
        mostrar_resumen=True,
        mostrar_head=True
    ):

        data_dir = Path(data_dir)
        patron = str(data_dir / "secop_chunk_*.csv")

        ddf = dd.read_csv(
            patron,
            dtype="object",
            assume_missing=True,
            blocksize=None
        )

        print("Dask DataFrame cargado correctamente.")

        if mostrar_resumen:

            print("\nInformación general")
            print("-" * 60)

            print("Número de particiones:", ddf.npartitions)

            print("\nFilas por partición estimadas:")
            print(
                ddf.map_partitions(len)
                .compute()
            )

            print("\nTipos de datos:")
            print(ddf.dtypes)

            print("\nColumnas detectadas:")
            print(ddf.columns.tolist())

            n_registros = ddf.shape[0].compute()

            print("\nTotal registros cargados:")
            print(f"{n_registros:,}")

        if mostrar_head:
            print("\nPrimeras observaciones:")
            print(ddf.head(10))

        return ddf

    # ========================================================
    # 5. NORMALIZACIÓN CANÓNICA DE COLUMNAS
    # ========================================================

    def normalizar_nombre_columna(self, col):

        col = str(col).strip().lower()

        col = (
            unicodedata.normalize("NFKD", col)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )

        col = re.sub(r"\s+", "_", col)
        col = re.sub(r"[^a-z0-9_]", "_", col)
        col = re.sub(r"_+", "_", col)
        col = col.strip("_")

        return col

    def normalizar_columnas(self, ddf):

        columnas_originales = ddf.columns.tolist()

        columnas_limpias = [
            self.normalizar_nombre_columna(col)
            for col in columnas_originales
        ]

        columnas_finales = []
        contador = {}

        for col in columnas_limpias:

            if col in contador:
                contador[col] += 1
                columnas_finales.append(f"{col}_{contador[col]}")
            else:
                contador[col] = 0
                columnas_finales.append(col)

        ddf.columns = columnas_finales

        print("Columnas normalizadas correctamente.")
        print("Columnas únicas:", len(set(ddf.columns)) == len(ddf.columns))

        return ddf

    # ========================================================
    # 6. RENOMBRADO SEMÁNTICO
    # ========================================================

    def renombrar_columnas_semanticas(self, ddf):

        diccionario_existente = {
            origen: destino
            for origen, destino in self.diccionario_semantico.items()
            if origen in ddf.columns
        }

        ddf = ddf.rename(columns=diccionario_existente)

        print("Columnas renombradas semánticamente:")

        for origen, destino in diccionario_existente.items():
            print(f"{origen} -> {destino}")

        return ddf

    # ========================================================
    # 7. NORMALIZACIÓN DE TEXTO
    # ========================================================

    def normalizar_texto(self, serie, valor_nulo="NO DEFINIDO"):

        return (
            serie
            .fillna(valor_nulo)
            .str.strip()
            .str.upper()
            .str.replace(r"\s+", " ", regex=True)
        )

    # ========================================================
    # 8. SELECCIÓN DE VARIABLES PREGUNTA 1
    # ========================================================

    def seleccionar_variables_pregunta_1(self, ddf):

        faltantes = [
            col for col in self.columnas_q1
            if col not in ddf.columns
        ]

        if faltantes:
            raise ValueError(
                f"Faltan columnas requeridas para Pregunta 1: {faltantes}"
            )

        ddf_q1_raw = ddf[self.columnas_q1].copy()

        print("DataFrame auxiliar Pregunta 1 creado.")
        print("Columnas:", ddf_q1_raw.columns.tolist())

        return ddf_q1_raw

    # ========================================================
    # 9. LIMPIEZA Y TRANSFORMACIÓN PREGUNTA 1
    # ========================================================

    def limpiar_pregunta_1(self, ddf_q1_raw):

        ddf_q1 = ddf_q1_raw.copy()

        ddf_q1["departamento"] = self.normalizar_texto(
            ddf_q1["departamento"]
        )

        ddf_q1["estado_adjudicacion"] = self.normalizar_texto(
            ddf_q1["estado_adjudicacion"]
        )

        ddf_q1["valor_adjudicado"] = dd.to_numeric(
            ddf_q1["valor_adjudicado"],
            errors="coerce"
        )

        ddf_q1["fecha_publicacion"] = dd.to_datetime(
            ddf_q1["fecha_publicacion"],
            errors="coerce"
        )

        ddf_q1["anio_publicacion"] = (
            ddf_q1["fecha_publicacion"]
            .dt.year
        )

        print("Limpieza y transformación Pregunta 1 finalizada.")

        return ddf_q1

    # ========================================================
    # 10. DIAGNÓSTICO DE CALIDAD PREGUNTA 1
    # ========================================================

    def diagnosticar_pregunta_1(self, ddf_q1):
        """
        Genera diagnóstico de calidad para variables usadas
        en la pregunta 1.
        """

        print("\n" + "=" * 70)
        print("DIAGNÓSTICO DE CALIDAD — PREGUNTA 1")
        print("=" * 70)

        diagnostico = {
            "total_registros":
                ddf_q1.shape[0],

            "nulos_valor_adjudicado":
                ddf_q1["valor_adjudicado"].isna().sum(),

            "valores_cero":
                (ddf_q1["valor_adjudicado"] == 0).sum(),

            "valores_negativos":
                (ddf_q1["valor_adjudicado"] < 0).sum(),

            "fechas_invalidas":
                ddf_q1["fecha_publicacion"].isna().sum(),

            "anios_invalidos":
                ddf_q1["anio_publicacion"].isna().sum(),

            "departamento_no_definido":
                (
                    ddf_q1["departamento"] == "NO DEFINIDO"
                ).sum(),

            "procesos_adjudicados_si":
                (
                    ddf_q1["estado_adjudicacion"] == "SI"
                ).sum(),

            "procesos_adjudicados_no":
                (
                    ddf_q1["estado_adjudicacion"] == "NO"
                ).sum()
        }

        diagnostico = dd.compute(diagnostico)[0]

        diagnostico_df = pd.DataFrame(
            list(diagnostico.items()),
            columns=[
                "Indicador",
                "Valor"
            ]
        )

        return diagnostico_df

    # ========================================================
    # 11. UNIVERSO ANALÍTICO PREGUNTA 1
    # ========================================================

    def crear_universo_analitico_q1(self, ddf_q1):

        ddf_q1_analitico = ddf_q1[
            (ddf_q1["estado_adjudicacion"] == "SI") &
            (ddf_q1["valor_adjudicado"] > 0) &
            (ddf_q1["departamento"] != "NO DEFINIDO") &
            (ddf_q1["anio_publicacion"].notnull())
        ]

        print("Universo analítico Pregunta 1 creado.")

        return ddf_q1_analitico