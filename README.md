# Proyecto Final de Computación Gráfica para Ingenieros

Este proyecto consiste en modelar y gráficar un conjunto de naves espaciales que son capaces 
de moverse libremente en el espacio, las librerias usadas fueron OpenGL y Pyglet para gráficar, 
Numpy y Math para ejecutar cálculos sobre conjuntos de elementos, lo restante tal como generar
curvas, cargar las Shapes en la GPU, cargar particulas en la GPU, grafo de escena y matrices de 
transformaciones fueron entregadas por el profesor y auxiliares del ramo "Computación Gráfica para 
Ingenieros" dictado en el semestre de otoño del año 2023, el cual pertenece a la malla de 
Ingeniería Civil en Computación de la Universidad de Chile.

A parte de lo mencioando este proyecto fue extendido implementando:

1 - Grabación y reproducción de movimiento.
2 - Modelo de iluminación de Phong con sus respectivas constantes modificables por el usuario.
3 - Cámara en tercera persona.
4 - Zoom variable en ambas vistas.
5 - Música para ambientar los distintos modos.

Todos los controles se especifican aquí:

- 'W' y 'S' para avanzar hacia adelante y atras respectivamente.
- 'A' y 'D' para rotar hacia la izquierda y derecha respectivamente.
- "Mouse hacia arriba" y "Mouse hacia abajo" para rotar hacia abajo y hacia arriba respectivamente.
- 'Q' para activar y desactivar la aparición de naves secundarias.
- 'E' para restablecer orientación de la/s nave/s.
- 'F' para activar y desactivar pantalla completa.

- 'R' para grabar posición actual, generando una curva en el espacio cuando hayan más de dos puntos guardados.
- '1' para que la nave recorra la curva grabada.
- 'V' para activar y desactivar la visibilidad de la curva.
- 'TAB' para eliminar la curva ya grabada.

- 'SPACE' para activar y desactivar el módelo de Iluminación de Phong.
- 'T' y 'G' para aumentar y disminuir el valor de la constante en el polinomio que describe la iluminación de Phong.
- 'Y' y 'H' para aumentar y disminuir el valor de la constante Lineal en el polinomio que describe la iluminación de Phong.
- 'U' y 'J' para aumentar y disminuir el valor de la constante Cuadrática en el polinomio que describe la iluminación de Phong.
- 'I' y 'K' para aumentar y disminuir la componente Ambiental del Modelo de iluminación de Phong.
- 'O' y 'L' para aumentar y disminuir la componente Difusa del Modelo de iluminación de Phong.
- 'P' y '-' para aumentar y disminuir la componente Especular del Modelo de iluminación de Phong.
- 'N' para ir cambiando entre distintas combinaciones de los parametros mencionados anteriormente,
(Existen combinaciones con flashes bastantes dislumbrantes, en caso de epilepsia tener precaución).

- 'C' para activar y desactivar cámara en tercera persona.
- 'X' para acercar zoom en ambas vistas.
- 'Z' para alejar zoom en ambas vistas.

Se debe tener en cuenta que la música se reproducirá una vez abierto el archivo .py por lo que se recomienda bajar el volumen
en caso de tener este mismo alto.

Para finalizar, el modelo 3D de las naves fue hecho por mi y las canciones ocupadas pertenecen al juego llamado
"The End is Nigh", compuestas por la banda "Ridiculon".

Espero que les guste este proyecto ;) y puedan jugar con los parametros de la iluminación de Phong, hay muchas combinaciones interesantes :D.
