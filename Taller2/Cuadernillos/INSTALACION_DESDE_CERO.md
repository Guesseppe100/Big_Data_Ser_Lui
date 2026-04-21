# Instalacion desde cero (Big Data Course)

Este documento explica la instalacion completa para una persona que inicia desde cero.

## Objetivo

Dejar funcionando:

- Docker
- stack del curso por modulos
- Git en contenedor para flujo guiado

## 1. Requisitos minimos

- CPU con virtualizacion habilitada
- 8 GB RAM (recomendado 16 GB)
- 10 GB de disco libre
- internet estable para descarga de imagenes

## 2. Instalar Docker

### Windows 10/11

1. Habilitar virtualizacion en BIOS/UEFI.
2. Instalar WSL2 en PowerShell (admin):

```powershell
wsl --install -d Ubuntu
```

3. Reiniciar equipo.
4. Instalar Docker Desktop desde:
   `https://www.docker.com/products/docker-desktop/`
5. Abrir Docker Desktop y esperar estado `Engine running`.

### macOS

1. Instalar Docker Desktop para macOS desde el sitio oficial.
2. Abrir Docker Desktop y autorizar permisos requeridos.
3. Confirmar que Docker Engine este activo.

### Linux (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
```

Cierra sesion y vuelve a entrar para aplicar el grupo `docker`.

## 3. Verificar instalacion

```bash
docker --version
docker compose version
docker run --rm hello-world
```

## 4. Configurar proyecto

Desde `C:\Users\nib1l\Documents\bigdata-course`:

```bash
cp .env.example .env
```

Si usas PowerShell en Windows:

```powershell
Copy-Item .env.example .env
```

## 5. Levantar servicios por modulo

### Inicio recomendado (primeras clases)

```bash
docker compose --profile airflow --profile dask up -d
```

### Verificar estado

```bash
docker compose ps
```

### Validar herramientas

```bash
docker compose exec git git --version
docker compose exec airflow-webserver airflow version
docker compose exec dask-scheduler dask --version
```

## 6. Interfaces utiles

- Airflow: `http://localhost:8080`
- Dask Dashboard: `http://localhost:8787`
- Spark UI: `http://localhost:8081`
- Jupyter: `http://localhost:8888`
- Kibana: `http://localhost:5601`

## 7. Problemas comunes

### No conecta a Docker daemon

Solucion:

- iniciar Docker Desktop
- esperar motor activo
- reintentar comando

### `docker compose` no existe

Solucion:

- actualizar Docker Desktop
- en Linux instalar `docker-compose-plugin`

### Falta memoria RAM

Solucion:

- levantar solo perfiles del modulo actual
- evitar `search` y `notebooks` cuando no se usen
- apagar todo antes de cambiar de modulo:

```bash
docker compose down
```

## 8. Regla operativa del curso

En este curso no se levanta todo el stack siempre.
Se levanta solo lo que necesita el modulo activo.
