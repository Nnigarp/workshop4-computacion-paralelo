#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WORKSHOP 4 - PARALLEL SUM OF SQUARES
Thread Synchronization with Mutexes

Objetivo:
    Comparar una implementación secuencial y una implementación paralela
    con threads para calcular la suma de cuadrados de un rango definido por
    start, end y step.

Criterios aplicados:
    1. Usar, en la medida de lo posible, todos los cores disponibles.
    2. Las versiones secuencial y paralela procesan exactamente el mismo rango.
    3. No se introducen retrasos artificiales (no se usa sleep).
    4. Se valida que los resultados secuencial y paralelo sean idénticos.

La versión paralela usa threading.Thread y threading.Lock. Cada thread calcula
un subtotal local y actualiza una única suma compartida dentro de una sección
crítica protegida por mutex. De este modo se reduce la contención del Lock.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Iterable


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResultadoBenchmark:
    start: int
    end: int
    step: int
    elementos: int
    threads: int
    tiempo_secuencial: float
    tiempo_paralelo: float
    speedup: float
    eficiencia: float
    resultado_correcto: bool


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

class Reporte:
    """Imprime en consola y conserva exactamente el mismo texto para el TXT."""

    def __init__(self) -> None:
        self._lineas: list[str] = []

    def escribir(self, texto: str = "") -> None:
        print(texto)
        self._lineas.append(texto)

    def guardar(self, ruta: Path) -> None:
        ruta.write_text("\n".join(self._lineas) + "\n", encoding="utf-8")


def linea(char: str = "=", ancho: int = 78) -> str:
    return char * ancho


def promedio(valores: Iterable[float]) -> float:
    valores = list(valores)
    if not valores:
        raise ValueError("No hay valores para calcular el promedio.")
    return statistics.fmean(valores)


def threads_por_defecto() -> list[int]:
    """
    Genera 1, 2, 4, 8, ... hasta alcanzar la cantidad de procesadores
    lógicos disponibles. Si el total no es potencia de dos, también se incluye
    explícitamente ese máximo.
    """
    logicos = os.cpu_count() or 1
    valores = [1]
    actual = 2

    while actual < logicos:
        valores.append(actual)
        actual *= 2

    if logicos not in valores:
        valores.append(logicos)

    return valores


def quitar_duplicados_preservando_orden(valores: Iterable[int]) -> list[int]:
    vistos: set[int] = set()
    salida: list[int] = []

    for valor in valores:
        if valor not in vistos:
            vistos.add(valor)
            salida.append(valor)

    return salida


# ---------------------------------------------------------------------------
# Validación de entrada
# ---------------------------------------------------------------------------

def validar_parametros(
    start: int,
    ends: list[int],
    step: int,
    threads: list[int],
    repeticiones: int,
) -> None:
    """Valida todas las entradas antes de iniciar cualquier benchmark."""

    if step == 0:
        raise ValueError("step no puede ser 0.")

    if not ends:
        raise ValueError("Debe especificarse al menos un valor para end.")

    if repeticiones <= 0:
        raise ValueError("El número de repeticiones debe ser mayor que 0.")

    if not threads:
        raise ValueError(
            "Debe especificarse al menos una cantidad de threads."
        )

    if any(t <= 0 for t in threads):
        raise ValueError(
            "Todas las cantidades de threads deben ser mayores que 0."
        )

    for end in ends:
        r = range(start, end, step)
        if len(r) == 0:
            raise ValueError(
                f"El rango range({start}, {end}, {step}) está vacío. "
                "Revise start, end y step."
            )


# ---------------------------------------------------------------------------
# Implementación secuencial
# ---------------------------------------------------------------------------

def suma_cuadrados_secuencial(start: int, end: int, step: int) -> int:
    """
    Calcula secuencialmente:
        sum(x*x for x in range(start, end, step))
    """
    total = 0

    for numero in range(start, end, step):
        total += numero * numero

    return total


# ---------------------------------------------------------------------------
# Partición del trabajo
# ---------------------------------------------------------------------------

def dividir_rango_equilibrado(
    rango: range,
    num_threads: int,
) -> list[range]:
    """
    Divide un range en num_threads subrangos cuyos tamaños difieren como máximo
    en un elemento. No materializa el rango completo en memoria.

    Garantías:
        - No se pierde ningún elemento.
        - No se duplica ningún elemento.
        - La unión de los subrangos equivale exactamente al rango original.
    """
    n = len(rango)
    base, resto = divmod(n, num_threads)

    particiones: list[range] = []
    indice = 0

    for i in range(num_threads):
        cantidad = base + (1 if i < resto else 0)
        siguiente = indice + cantidad
        particiones.append(rango[indice:siguiente])
        indice = siguiente

    if sum(len(p) for p in particiones) != n:
        raise RuntimeError(
            "Error interno: la partición no conserva el trabajo total."
        )

    return particiones


# ---------------------------------------------------------------------------
# Implementación paralela con Mutex
# ---------------------------------------------------------------------------

def suma_cuadrados_paralela(
    start: int,
    end: int,
    step: int,
    num_threads: int,
) -> tuple[int, list[int]]:
    """
    Calcula la suma de cuadrados en paralelo utilizando Thread + Lock.

    Diseño de sincronización:
        1. Cada thread procesa exclusivamente su subrango.
        2. Cada thread acumula un subtotal LOCAL, sin necesidad de mutex.
        3. La única actualización al total COMPARTIDO se realiza dentro de:
               with mutex:
                   total_compartido += subtotal
        4. El hilo principal espera con join(); no se usa sleep().
    """
    rango = range(start, end, step)
    particiones = dividir_rango_equilibrado(rango, num_threads)

    total_compartido = 0
    mutex = Lock()

    def trabajador(subrango: range) -> None:
        nonlocal total_compartido

        subtotal = 0
        for numero in subrango:
            subtotal += numero * numero

        # Sección crítica mínima: únicamente la actualización compartida.
        with mutex:
            total_compartido += subtotal

    hilos = [
        Thread(
            target=trabajador,
            args=(subrango,),
            name=f"worker-{i + 1}",
        )
        for i, subrango in enumerate(particiones)
    ]

    for hilo in hilos:
        hilo.start()

    # Sin retrasos artificiales: se espera correctamente a cada thread.
    for hilo in hilos:
        hilo.join()

    cargas = [len(p) for p in particiones]
    return total_compartido, cargas


# ---------------------------------------------------------------------------
# Medición
# ---------------------------------------------------------------------------

def medir_secuencial(
    start: int,
    end: int,
    step: int,
    repeticiones: int,
    reporte: Reporte,
) -> tuple[int, list[float]]:
    tiempos: list[float] = []
    resultado_referencia: int | None = None

    for repeticion in range(1, repeticiones + 1):
        inicio = time.perf_counter()
        resultado = suma_cuadrados_secuencial(start, end, step)
        fin = time.perf_counter()

        duracion = fin - inicio
        tiempos.append(duracion)

        if resultado_referencia is None:
            resultado_referencia = resultado
        elif resultado != resultado_referencia:
            raise RuntimeError(
                "La implementación secuencial produjo resultados "
                "inconsistentes."
            )

        reporte.escribir(
            f"Repetición {repeticion:>2}: {duracion:.9f} s"
        )

    assert resultado_referencia is not None
    return resultado_referencia, tiempos


def medir_paralela(
    start: int,
    end: int,
    step: int,
    num_threads: int,
    repeticiones: int,
    resultado_esperado: int,
    elementos_esperados: int,
    reporte: Reporte,
) -> tuple[int, list[float], list[int]]:
    tiempos: list[float] = []
    resultado_referencia: int | None = None
    cargas_referencia: list[int] | None = None

    for repeticion in range(1, repeticiones + 1):
        inicio = time.perf_counter()
        resultado, cargas = suma_cuadrados_paralela(
            start, end, step, num_threads
        )
        fin = time.perf_counter()

        duracion = fin - inicio
        tiempos.append(duracion)

        # Regla crítica: serial y paralelo deben producir exactamente lo mismo.
        if resultado != resultado_esperado:
            raise RuntimeError(
                "\nERROR DE VALIDACIÓN\n"
                f"Resultado secuencial : {resultado_esperado}\n"
                f"Resultado paralelo   : {resultado}\n"
                "La medición se aborta porque los resultados no coinciden."
            )

        # Regla crítica: la versión paralela debe procesar exactamente
        # la misma cantidad de elementos que la secuencial.
        if sum(cargas) != elementos_esperados:
            raise RuntimeError(
                "La versión paralela no procesó la misma cantidad de trabajo."
            )

        if resultado_referencia is None:
            resultado_referencia = resultado
            cargas_referencia = cargas
        elif resultado != resultado_referencia:
            raise RuntimeError(
                "La implementación paralela produjo resultados inconsistentes."
            )

        reporte.escribir(
            f"Repetición {repeticion:>2}: {duracion:.9f} s | "
            f"Validación: CORRECTA"
        )

    assert resultado_referencia is not None
    assert cargas_referencia is not None
    return resultado_referencia, tiempos, cargas_referencia


# ---------------------------------------------------------------------------
# Presentación y persistencia
# ---------------------------------------------------------------------------

def mostrar_cargas(cargas: list[int], reporte: Reporte) -> None:
    minimo = min(cargas)
    maximo = max(cargas)

    reporte.escribir(
        f"Distribución de trabajo : mín. {minimo} | "
        f"máx. {maximo} elementos/thread"
    )
    reporte.escribir(
        f"Diferencia máxima       : {maximo - minimo} elemento(s)"
    )


def guardar_csv(resultados: list[ResultadoBenchmark], ruta: Path) -> None:
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(
            [
                "start",
                "end",
                "step",
                "elementos",
                "threads",
                "tiempo_secuencial_s",
                "tiempo_paralelo_s",
                "speedup",
                "eficiencia_pct",
                "resultado_correcto",
            ]
        )

        for r in resultados:
            writer.writerow(
                [
                    r.start,
                    r.end,
                    r.step,
                    r.elementos,
                    r.threads,
                    f"{r.tiempo_secuencial:.9f}",
                    f"{r.tiempo_paralelo:.9f}",
                    f"{r.speedup:.6f}",
                    f"{r.eficiencia:.4f}",
                    "SI" if r.resultado_correcto else "NO",
                ]
            )


def mostrar_tabla_resumen(
    resultados: list[ResultadoBenchmark],
    reporte: Reporte,
) -> None:
    reporte.escribir()
    reporte.escribir(linea())
    reporte.escribir("RESUMEN COMPARATIVO")
    reporte.escribir(linea())

    encabezado = (
        f"{'Elementos':>12} "
        f"{'Threads':>8} "
        f"{'T. Sec (s)':>12} "
        f"{'T. Par (s)':>12} "
        f"{'Speedup':>10} "
        f"{'Efic. %':>10} "
        f"{'OK':>5}"
    )
    reporte.escribir(encabezado)
    reporte.escribir("-" * len(encabezado))

    for r in resultados:
        reporte.escribir(
            f"{r.elementos:>12} "
            f"{r.threads:>8} "
            f"{r.tiempo_secuencial:>12.6f} "
            f"{r.tiempo_paralelo:>12.6f} "
            f"{r.speedup:>9.4f}x "
            f"{r.eficiencia:>10.2f} "
            f"{'SI':>5}"
        )


# ---------------------------------------------------------------------------
# Benchmark completo
# ---------------------------------------------------------------------------

def ejecutar_benchmark(args: argparse.Namespace) -> int:
    ends = quitar_duplicados_preservando_orden(args.ends)
    threads = quitar_duplicados_preservando_orden(args.threads)

    validar_parametros(
        start=args.start,
        ends=ends,
        step=args.step,
        threads=threads,
        repeticiones=args.repeticiones,
    )

    reporte = Reporte()
    resultados: list[ResultadoBenchmark] = []

    cpu_logicos = os.cpu_count() or 1

    reporte.escribir(linea())
    reporte.escribir("WORKSHOP 4 - PARALLEL SUM OF SQUARES")
    reporte.escribir("THREAD SYNCHRONIZATION WITH MUTEXES")
    reporte.escribir(linea())
    reporte.escribir()
    reporte.escribir(
        f"Sistema operativo      : {platform.system()} "
        f"{platform.release()}"
    )
    reporte.escribir(f"Python                 : {platform.python_version()}")
    reporte.escribir(f"Procesadores lógicos   : {cpu_logicos}")
    reporte.escribir(f"Threads evaluados      : {threads}")
    reporte.escribir(f"Repeticiones           : {args.repeticiones}")
    reporte.escribir()
    reporte.escribir("REGLAS CRÍTICAS DEL CURSO")
    reporte.escribir(
        "  [OK] Mismo rango y misma cantidad de elementos en "
        "serial/paralelo"
    )
    reporte.escribir("  [OK] Sin retrasos artificiales: no se utiliza sleep()")
    reporte.escribir("  [OK] Validación obligatoria de igualdad de resultados")
    if max(threads) >= cpu_logicos:
        reporte.escribir(
            "  [OK] Se evalúa la capacidad lógica disponible "
            f"({cpu_logicos} threads)"
        )
    else:
        reporte.escribir(
            f"  [AVISO] Máximo solicitado: {max(threads)}; "
            f"capacidad lógica disponible: {cpu_logicos}"
        )

    for end in ends:
        rango = range(args.start, end, args.step)
        elementos = len(rango)

        reporte.escribir()
        reporte.escribir(linea())
        reporte.escribir(
            f"RANGO: range({args.start}, {end}, {args.step})"
        )
        reporte.escribir(linea())
        reporte.escribir(f"Cantidad de elementos : {elementos}")

        # ------------------------- Secuencial -------------------------
        reporte.escribir()
        reporte.escribir("EJECUCIÓN SECUENCIAL")
        reporte.escribir("-" * 78)

        resultado_seq, tiempos_seq = medir_secuencial(
            args.start,
            end,
            args.step,
            args.repeticiones,
            reporte,
        )
        promedio_seq = promedio(tiempos_seq)

        reporte.escribir(f"Promedio secuencial    : {promedio_seq:.9f} s")
        reporte.escribir(f"Resultado secuencial   : {resultado_seq}")

        # -------------------------- Paralelo --------------------------
        for num_threads in threads:
            reporte.escribir()
            reporte.escribir(
                f"EJECUCIÓN PARALELA - {num_threads} THREAD(S)"
            )
            reporte.escribir("-" * 78)

            resultado_par, tiempos_par, cargas = medir_paralela(
                args.start,
                end,
                args.step,
                num_threads,
                args.repeticiones,
                resultado_seq,
                elementos,
                reporte,
            )

            promedio_par = promedio(tiempos_par)
            speedup = (
                promedio_seq / promedio_par
                if promedio_par > 0
                else float("inf")
            )
            eficiencia = (speedup / num_threads) * 100.0

            mostrar_cargas(cargas, reporte)
            reporte.escribir(f"Promedio paralelo      : {promedio_par:.9f} s")
            reporte.escribir(f"Resultado paralelo     : {resultado_par}")
            reporte.escribir("Resultados idénticos   : SÍ")
            reporte.escribir(f"Speedup                : {speedup:.6f}x")
            reporte.escribir(f"Eficiencia             : {eficiencia:.2f} %")

            resultados.append(
                ResultadoBenchmark(
                    start=args.start,
                    end=end,
                    step=args.step,
                    elementos=elementos,
                    threads=num_threads,
                    tiempo_secuencial=promedio_seq,
                    tiempo_paralelo=promedio_par,
                    speedup=speedup,
                    eficiencia=eficiencia,
                    resultado_correcto=True,
                )
            )

    mostrar_tabla_resumen(resultados, reporte)

    reporte.escribir()
    reporte.escribir(linea())
    reporte.escribir("VALIDACIÓN GENERAL: CORRECTA")
    reporte.escribir(linea())
    reporte.escribir(
        "Todas las configuraciones paralelas procesaron el mismo rango que "
        "la versión secuencial y produjeron exactamente el mismo resultado."
    )
    reporte.escribir(
        "Nota: en CPython, este cálculo es CPU-bound; por el GIL y "
        "el overhead "
        "de threads/sincronización, un mayor número de threads no garantiza "
        "speedup > 1. Los resultados medidos deben analizarse tal "
        "como ocurran."
    )

    directorio = Path(args.salida)
    directorio.mkdir(parents=True, exist_ok=True)

    ruta_txt = directorio / "resultados_workshop4.txt"
    ruta_csv = directorio / "resultados_workshop4.csv"

    reporte.guardar(ruta_txt)
    guardar_csv(resultados, ruta_csv)

    print()
    print(f"TXT guardado en : {ruta_txt.resolve()}")
    print(f"CSV guardado en : {ruta_csv.resolve()}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Workshop 4: suma de cuadrados secuencial vs paralela usando "
            "threads y mutexes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Inicio común de los rangos.",
    )
    parser.add_argument(
        "--ends",
        type=int,
        nargs="+",
        default=[100_000, 1_000_000, 5_000_000],
        help="Uno o más valores end para experimentar con diferentes rangos.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Paso del range. No puede ser 0.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=threads_por_defecto(),
        help=(
            "Cantidades de threads a evaluar. Por defecto se prueban "
            "potencias "
            "de 2 y también el máximo de procesadores lógicos disponibles."
        ),
    )
    parser.add_argument(
        "--repeticiones",
        type=int,
        default=3,
        help="Número de repeticiones por configuración.",
    )
    parser.add_argument(
        "--salida",
        type=str,
        default="resultados_workshop4",
        help="Directorio donde se guardarán el TXT y CSV.",
    )

    return parser


def main() -> int:
    parser = crear_parser()
    args = parser.parse_args()

    try:
        return ejecutar_benchmark(args)
    except (ValueError, RuntimeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nEjecución cancelada por el usuario.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
