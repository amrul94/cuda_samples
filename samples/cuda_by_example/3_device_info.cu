//
// Created by amrulla on 18.04.2024.
//

#include "common/error_handling.h"

int main() {
    cudaDeviceProp prop{};
    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));

    for (int i = 0; i < count; i++) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        printf(" --- Общая информация об устройстве %d ---\n", i);
        printf("Имя: %s\n", prop.name);
        printf("Вычислительные возможности: %d.%d\n", prop.major, prop.minor);
        printf("Тактовая частота: %d\n", prop.clockRate);
        printf("Перекрытие копирования: ");
        if (prop.deviceOverlap)
            printf("Разрешено\n");
        else
            printf("Запрещено\n");
        printf("Тайм-аут выполнения ядра : ");
        if (prop.kernelExecTimeoutEnabled)
            printf("Включен\n\n");
        else
            printf("Выключен\n\n");

        printf(" --- Информация о памяти для устройства %d ---\n", i);
        printf("Всего глобальной памяти: %ld\n", prop.totalGlobalMem);
        printf("Всего константной памяти: %ld\n", prop.totalConstMem);
        printf("Максимальный шаг: %ld\n", prop.memPitch);
        printf("Выравнивание текстур: %ld\n\n", prop.textureAlignment);

        printf(" --- Информация о мультипроцессорах для устройства %d ---\n", i);
        printf("Количество мультипроцессоров: %d\n",
               prop.multiProcessorCount);
        printf("Разделяемая память на один МП: %ld\n", prop.sharedMemPerBlock);
        printf("Регистров на один МП: %d\n", prop.regsPerBlock);
        printf("Нитей в варпе: %d\n", prop.warpSize);
        printf("Макс. количество нитей в блоке: %d\n",
               prop.maxThreadsPerBlock);
        printf("Макс. количество нитей по измерениям: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
        printf("Максимальные размеры сетки: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1],
               prop.maxGridSize[2]);
        printf("\n");
    }
}