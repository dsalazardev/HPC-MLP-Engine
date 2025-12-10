"""
MLP con Multiprocessing - Benchmark de Speedup

Ejecuta el entrenamiento con 1, 2, 4, 8, 16 procesos
y genera datos para la gráfica de speedup.

Speedup(p) = T(1) / T(p)
Eficiencia(p) = Speedup(p) / p
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from Paralelo.Multiprocessing.src.DataLoader import DataLoader
from Paralelo.Multiprocessing.src.MLP import MLP
from pathlib import Path


def run_training(X_train, y_train, X_test, y_test_raw, n_workers, layer_structure, lr, epochs, batch_size):
    """Ejecuta entrenamiento con n_workers y retorna (tiempo, accuracy)"""
    mlp = MLP(layer_structure, lr, n_workers=n_workers)
    
    start_time = time.time()
    mlp.train(X_train, y_train, epochs, batch_size)
    end_time = time.time()
    
    predictions = mlp.predict(X_test)
    accuracy = np.mean(predictions == y_test_raw)
    
    return end_time - start_time, accuracy


def main():
    # =================================================================
    # CONFIGURACIÓN
    # =================================================================
    EPOCHS = 10
    BATCH_SIZE = 250
    LEARNING_RATE = 0.1
    LAYER_STRUCTURE = [784, 250, 10]
    WORKER_COUNTS = [1, 2, 4, 8, 16]  # Procesos a probar
    
    DATASET_PATH = str(Path(__file__).resolve().parents[2] / 'Dataset' / 'archive')
    
    print("=" * 70)
    print("  BENCHMARK DE SPEEDUP - MLP CON MULTIPROCESSING")
    print("=" * 70)
    print(f"CPUs disponibles: {cpu_count()}")
    print(f"Arquitectura: {LAYER_STRUCTURE}")
    print(f"Épocas: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"Procesos a probar: {WORKER_COUNTS}")
    
    # =================================================================
    # CARGA DE DATOS
    # =================================================================
    print("\nCargando dataset MNIST...")
    loader = DataLoader(DATASET_PATH)
    (X_train, y_train_raw), (X_test, y_test_raw) = loader.load_mnist()
    y_train_encoded = loader.one_hot_encode(y_train_raw, 10)
    
    print(f"  -> Training: {X_train.shape}")
    print(f"  -> Test: {X_test.shape}")
    
    # =================================================================
    # BENCHMARK CON DIFERENTES NÚMEROS DE PROCESOS
    # =================================================================
    results = []
    
    print("\n" + "=" * 70)
    print("  EJECUTANDO BENCHMARKS")
    print("=" * 70)
    
    for n_workers in WORKER_COUNTS:
        print(f"\n{'='*70}")
        print(f"  PROBANDO CON {n_workers} PROCESO(S)")
        print(f"{'='*70}")
        
        tiempo, accuracy = run_training(
            X_train, y_train_encoded, X_test, y_test_raw,
            n_workers=n_workers,
            layer_structure=LAYER_STRUCTURE,
            lr=LEARNING_RATE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        
        results.append({
            'workers': n_workers,
            'tiempo': tiempo,
            'accuracy': accuracy
        })
        
        print(f"\n>>> Resultado: {tiempo:.2f}s | Accuracy: {accuracy*100:.2f}%")
    
    # =================================================================
    # CALCULAR SPEEDUP Y EFICIENCIA
    # =================================================================
    t1 = results[0]['tiempo']  # Tiempo con 1 proceso (base)
    
    print("\n" + "=" * 70)
    print("  RESULTADOS PARA GRÁFICA DE SPEEDUP")
    print("=" * 70)
    print(f"\nTiempo base T(1) = {t1:.2f} segundos")
    
    print("\n{:^10} | {:^12} | {:^10} | {:^12} | {:^12}".format(
        "Procesos", "Tiempo (s)", "Accuracy", "Speedup", "Eficiencia"
    ))
    print("-" * 70)
    
    for r in results:
        p = r['workers']
        tp = r['tiempo']
        acc = r['accuracy']
        speedup = t1 / tp
        eficiencia = speedup / p
        
        print("{:^10} | {:^12.2f} | {:^10.2f}% | {:^12.4f} | {:^12.2f}%".format(
            p, tp, acc * 100, speedup, eficiencia * 100
        ))
    
    # =================================================================
    # DATOS PARA COPIAR/PEGAR
    # =================================================================
    print("\n" + "=" * 70)
    print("  DATOS PARA COPIAR (formato CSV)")
    print("=" * 70)
    print("\nProcesos,Tiempo(s),Accuracy(%),Speedup,Eficiencia(%)")
    
    # Preparar datos para gráficas
    procesos = []
    tiempos = []
    speedups = []
    eficiencias = []
    
    for r in results:
        p = r['workers']
        tp = r['tiempo']
        acc = r['accuracy']
        speedup = t1 / tp
        eficiencia = speedup / p
        print(f"{p},{tp:.2f},{acc*100:.2f},{speedup:.4f},{eficiencia*100:.2f}")
        
        procesos.append(p)
        tiempos.append(tp)
        speedups.append(speedup)
        eficiencias.append(eficiencia * 100)
    
    # =================================================================
    # GENERAR GRÁFICAS
    # =================================================================
    print("\n" + "=" * 70)
    print("  GENERANDO GRÁFICAS")
    print("=" * 70)
    
    output_dir = Path(__file__).parent
    
    # Gráfica 1: Speedup
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(procesos, speedups, 'bo-', linewidth=2, markersize=10, label='Speedup Real')
    ax1.plot(procesos, procesos, 'r--', linewidth=2, label='Speedup Ideal (lineal)')
    ax1.set_xlabel('Número de Procesos', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('Speedup vs Número de Procesos\nMLP con Multiprocessing (Python)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(procesos)
    ax1.set_xlim(0, max(procesos) + 1)
    ax1.set_ylim(0, max(max(procesos), max(speedups)) + 1)
    
    speedup_path = output_dir / 'grafica_speedup.png'
    fig1.savefig(speedup_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Gráfica Speedup guardada: {speedup_path}")
    
    # Gráfica 2: Eficiencia
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(procesos, eficiencias, color='steelblue', edgecolor='navy', alpha=0.8)
    ax2.axhline(y=100, color='r', linestyle='--', linewidth=2, label='Eficiencia Ideal (100%)')
    ax2.set_xlabel('Número de Procesos', fontsize=12)
    ax2.set_ylabel('Eficiencia (%)', fontsize=12)
    ax2.set_title('Eficiencia vs Número de Procesos\nMLP con Multiprocessing (Python)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(procesos)
    ax2.set_ylim(0, 120)
    
    # Agregar valores en las barras
    for i, (p, e) in enumerate(zip(procesos, eficiencias)):
        ax2.text(p, e + 2, f'{e:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    eficiencia_path = output_dir / 'grafica_eficiencia.png'
    fig2.savefig(eficiencia_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Gráfica Eficiencia guardada: {eficiencia_path}")
    
    # Gráfica 3: Tiempo de Ejecución
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(procesos, tiempos, 'go-', linewidth=2, markersize=10)
    ax3.set_xlabel('Número de Procesos', fontsize=12)
    ax3.set_ylabel('Tiempo (segundos)', fontsize=12)
    ax3.set_title('Tiempo de Ejecución vs Número de Procesos\nMLP con Multiprocessing (Python)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(procesos)
    
    # Agregar valores en los puntos
    for p, t in zip(procesos, tiempos):
        ax3.annotate(f'{t:.1f}s', (p, t), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=10)
    
    tiempo_path = output_dir / 'grafica_tiempo.png'
    fig3.savefig(tiempo_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Gráfica Tiempo guardada: {tiempo_path}")
    
    # Gráfica 4: Combinada (Speedup + Eficiencia)
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Speedup
    ax4a.plot(procesos, speedups, 'bo-', linewidth=2, markersize=10, label='Speedup Real')
    ax4a.plot(procesos, procesos, 'r--', linewidth=2, label='Speedup Ideal')
    ax4a.set_xlabel('Número de Procesos', fontsize=11)
    ax4a.set_ylabel('Speedup', fontsize=11)
    ax4a.set_title('Speedup', fontsize=12)
    ax4a.legend()
    ax4a.grid(True, alpha=0.3)
    ax4a.set_xticks(procesos)
    
    # Eficiencia
    bars = ax4b.bar(procesos, eficiencias, color='steelblue', edgecolor='navy', alpha=0.8)
    ax4b.axhline(y=100, color='r', linestyle='--', linewidth=2, label='Ideal')
    ax4b.set_xlabel('Número de Procesos', fontsize=11)
    ax4b.set_ylabel('Eficiencia (%)', fontsize=11)
    ax4b.set_title('Eficiencia', fontsize=12)
    ax4b.legend()
    ax4b.grid(True, alpha=0.3, axis='y')
    ax4b.set_xticks(procesos)
    ax4b.set_ylim(0, 120)
    
    fig4.suptitle('Análisis de Rendimiento - MLP con Multiprocessing (Python)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    combinada_path = output_dir / 'grafica_combinada.png'
    fig4.savefig(combinada_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Gráfica Combinada guardada: {combinada_path}")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
