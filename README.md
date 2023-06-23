# Muscle Measure

## Requirements

* Docker. Please, note that `docker-compose` is needed too and is included in
the Docker Desktop installation. Docker Desktop is available for
[Mac](https://docs.docker.com/desktop/install/mac-install/),
[Windows](https://docs.docker.com/desktop/install/windows-install/) and
[Linux](https://docs.docker.com/desktop/install/linux-install/).

Note that you do not necessarily need to install Docker Desktop. You can also
install [Docker Engine](https://docs.docker.com/engine/install/) and
[docker-compose](https://docs.docker.com/compose/install/). In that case,
please use `docker-compose` instead of `docker compose` in all commands below.

* DVC, to manage the binary data like models and example images we use dvc to get data correctily, please follow instructions to install dvc [here](https://dvc.org/). 

## Getting started

The setup takes place in 3 steps.
1. Download the [zip data folder](https://drive.google.com/file/d/1h6Bq0jytUBLhi7TzByROGB1zNm7-4ZRr/view?usp=sharing) and decompress on the root of repository.
2. Use dvc command to get all necessary data.
```bash
$ dvc pull
```
box, but for development purposes, you might want to change some settings. The
`.env` includes settings for downloading/restoring the databases, running tests
etc. The `docker-compose.override.yml` includes settings for the ports, volumes
etc.
3. Start the containers and run the appropriate commands from below depending on
the way you want to run Joinup.

### 1.1 Environment variables

Copy the `.env.example` file as `.env` file in the root directory where you
should assign values for environment variables containing sensitive data, such
as passwords, tokens, hashes or even custom preferences. These variables are not
committed in VCS and due to their nature should be manually set.

More information about these environment variables can be found in the
`.env.dist` file.

### 1.2 Set up the docker-compose.override.yml file

Copy the `docker-compose.override.yml.dist` file as
`docker-compose.override.yml` file in the root directory. This file is not
committed in VCS and due to its nature should be manually set. Configuration
from `docker-compose.yml` can be overwritten in `docker-compose.override.yml`
file. Port mappings are not declared in the main `docker-compose.yml` file
because this is developer custom setting, so you'll need to set up yours in the
override file.

#### 1.2.1 Permissions

In order to be able to handle permissions properly, the correct arguments and/or
environment variables need to be passed to the containers. The
`docker-compose.override.yml` file contains the following settings:

```yaml
  web:
    image: joinup/web:latest
    build:
      context: ./resources/docker/web
#      args:
#        USER_ID: 501
#    environment:
#      DAEMON_GROUP: dialout
```
Out of the top, you will only need to keep 1 couple of settings from the
`USER_ID`, `GROUP_IS`, `DAEMON_USER` and `DAEMON_GROUP` variables. The idea is
that the host user will have to match the ID and GID of the DAEMON_USER and
DAEMON_GROUP that runs the web server. This is needed in order to be able to
write to the files and directories that are mounted in the container.

By default, `www-data` user is set with ID `33` and `www-data` group is set with
GID `33`. If you are using a Linux machine, and you are using the default user,
you should use the following settings:

```yaml
  web:
    image: joinup/web:latest
    build:
      context: ./resources/docker/web
      args:
        USER_ID: 1000
        GROUP_ID: 1000
```

If you are using a Mac, the default user ID and group ID are `501` and `20`
respectively. In that case, you need to use the following settings:

```yaml
  web:
    image: joinup/web:latest
    build:
      context: ./resources/docker/web
      args:
        USER_ID: 501
    environment:
      DAEMON_GROUP: dialout
```

For any other case, please, use the corresponding `USER_ID` and `GROUP_ID`
and/or the `DAEMON_USER` and `DAEMON_GROUP` variables.

```yaml
  web:
    image: joinup/web:latest
    build:
      context: ./resources/docker/web
      args:
        USER_ID: 1004
        GROUP_ID: 1004
```
or

```yaml
  web:
    image: joinup/web:latest
    build:
      context: ./resources/docker/web
    environment:
      DAEMON_USER: some_existing_user
      DAEMON_GROUP: some_existing_group
```

The variables can be used together but only one of `USER_ID` and `DAEMON_USER`
and only one of `GROUP_ID` and `DAEMON_GROUP` can be used at the same time.

### 1.3 Set up your local runner.yml.

Copy the `runner.yml.dist` file as `runner.yml` file in the root directory. The
file already contains 2 shortened commands for installing the codebase and
installing a clean website. You can add more commands in this file. The
`runner.yml` file is not committed in VCS and due to its nature should be
manually set.

See below section `Using the runner` for more information.

### 2.1 Personal directories

In order to persist the data between container restarts, the `volumes` entry in
the `docker-compose.override.yml` file is used. This entry is commented out by
default.

The following directories are needed to properly develop on docker:
* The `~/.ssh` directory is used to store the SSH keys for GitHub
authentication. If you want to manage the repositories with SSH, you need to
add your SSH keys in this directory.
* The `~/.composer` directory is the global configuration directory for
Composer. It is used to store the Composer cache and the authentication tokens.
* The `~/.npm` directory is used to store the NPM cache.

### 2.2 Hosts file

In order to access the website, you need to add the following line to your
`hosts` file:

```bash
127.0.0.1 web
```
This will allow you to access the website at http://web:8080/web.

The website can be accessible also at http://localhost:8080/web, but this will
not work for the Behat tests out of the box, neither for the commands ran by
drush by default.

### 3.1 Starting the containers

To start the containers, you can execute the following command from the project
root directory:

```bash
$ docker compose up --build --force-recreate -d
```

This will automatically read the `docker-compose.yml` file as a source. The `-d`
options will start the containers in the background. If you omit the `-d`
option, `docker compose` will run in the foreground.

### 3.2 Stopping the containers

To stop the containers, you can use the command `docker compose down` from the
same directory as the `docker-compose.yml`. Using this command, however, will
only stop the machine and will not destroy the volume that was created with it.
To clean the volume as well, use the `-v` parameter as `docker compose down -v`.

### 3.3 Handling XDEBUG

@todo Verify the bug that prevents XDEBUG from working properly when the
containers are started with the `DISABLE_XDEBUG` environment variable set to
`True`.

During development XDEBUG can be enabled or disabled by running the following:

```bash
# Enable
$ docker compose exec web phpenmod xdebug
# Disable
$ docker compose exec web phpdismod xdebug
```
The restart is needed in order to apply the settings in the PHP-fpm pool.

### 3.4 Installing the codebase

Run the following command to install all packages in the vendor folder:

```bash
$ docker compose exec --user www-data web composer install
```

### 3.5 Reset the environment

In order to restart the containers and reset the environment, you can run the
following command from the project root directory:

```bash
$ docker compose down -v && docker compose up -d --build --force-recreate
```

This will stop the containers, remove the volumes and start the containers
again.

### 3.6 Install an empty website

From the project root, run:

```bash
$ docker compose exec --user www-data web ./vendor/bin/run toolkit:install-clean
```
This will install a clean website with the default configuration. The website is
available at http://web:8080/web (Please, ensure you have added the `web` entry
in your `hosts` file as described in section 2.2).

### 3.7 Install a cloned site

A _cloned site_ install is also restoring the databases from the production and
is running all update scripts.

#### 3.7.1 Downloading the databases

Before proceeding, make sure you set manually the `NEXTCLOUD_*` environment variables in `.env` file. The values should be provided according to instructions provided [here](https://webgate.ec.europa.eu/fpfis/qa/how-to?nid=22351).

**Note:** After you have accessed Nextcloud, through CAS-Login button, and you have set up your account, you NEED to generate a new App name password/token and use the corresponding username/password credentials as `NEXTCLOUD_USER/NEXTCLOUD_PASS` environment variables.

Then, run the following command to download the databases:
```bash
$ docker compose exec --user www-data web ./vendor/bin/run dev:download-databases
```
Please, note that this requires the containers to be fired up. The databases
will be stored in the `./db/` directory.

#### 3.7.2 Restoring the databases

For docker, the databases are restored on container start. To do so, you need to
set the following environment variable in the `.env` file:
```
DOCKER_RESTORE_PRODUCTION=yes
```
After that, you will need to reset the environment, as described in the previous
section. **You will need to reset the environment every time you need to
reinstall the databases**. This is because the database is restored on container
start and the container is not restarted when you run the deploy commands.

Remember that in order to reset the environment, you need to run the following
command from the project root directory:
```bash
$ docker compose down -v && docker compose up -d --build --force-recreate
```
In order to stop the containers, remove the volumes and start the containers
again.

**Important**: The SQL database is restored from a dump file, which is not that
fast. The database container will take some time to complete the restoration and
you can monitor this process by running the following command:
```bash
$ docker compose logs -f mysql
```
Please, wait until the container reports that it is ready for connections before
proceeding to the deploy commands.

#### 3.7.3 Install the cloned site

After containers are up wait until the database is restored. In order to check
the status of the database, run the following command:
```bash
$ docker compose logs -f mysql
```
You will see the following message when the database is ready:
```
mysql_1  |  /docker-entrypoint.sh: running /docker-entrypoint-initdb.d/restore.sh
.
.
mysql_1  |  .........: /usr/sbin/mysqld: Shutdown complete .........
```
and after some time
```
mysql_1  |  .........: /usr/sbin/mysqld: ready for connections. Version .......
```
Then, you can proceed with the deploy commands.
```bash
# Sets up the settings for deployment.
$ docker compose exec --user www-data web ./vendor/bin/run drupal:settings install
# Runs the deployment.
$ docker compose exec --user www-data web ./vendor/bin/run toolkit:run-deploy
```

**Reminder**: If you want to switch between the cloned site and the empty site,
you will need to reset the environment, as described in the previous section.

#### 3.7.4 Install additional modules and users

Further modules are available to help with development. To install them, run:

```bash
# Disable the config-readonly.
$ docker compose exec --user www-data web ./vendor/bin/run config-readonly:disable

# Install the demo users.
$ docker compose exec --user www-data web ./vendor/bin/run dev:demo-users

# Install the additional modules.
$ docker compose exec --user www-data web ./vendor/bin/run dev:install-modules
```

A list of demo users will be installed in the website. The users are:
* "User": An authenticated user with email `user@example.com` and password
`user`.
* "Admin": An administrator user with email `admin@example.com` and password
`admin`.
* "Moderator": A moderator user with email `moderator@example.com` and password
`moderator`.
The `dev:demo-users` will also setup the CAS mock server which will allow you to
login bypassing EU Login.

The demo modules include views_ui, field_ui, and other helpful modules.

## Using the Task Runner

A Task Runner is available to help with the development. It is based on the
[Robo](https://robo.li/) library. The Task Runner is available in the container
`web` and can be executed with the following command:

```bash
$ docker compose exec --user www-data web ./vendor/bin/run
```
The above command will also list all the available commands.

You can generate your own commands by adding them to the `runner.yml` file in
the root directory. You can override and create your own commands there.

For example, the whole deployment command can be executed with the following
(note that this is supposed to replace the whole deploy procedure, including
the `drupal:settings` and `toolkit:run-deploy` commands - run this as the first
command after the environment is reset):

```yaml
commands:
  local-deploy:
    # Overrides the toolkit:install-clone command but does not install the
    # databases as they are restored in the container startup.
    - task: run
      command: joinup:environment
    - task: exec
      command: ${drush.bin} memcache:flush-all
    - task: run
      command: drupal:settings
      arguments:
        - install
    - task: run
      command: toolkit:run-deploy
    - task: run
      command: config-readonly:disable
    - task: run
      command: dev:demo-users
    - task: run
      command: dev:install-modules
    - task: run
      command: config-readonly:disable
    - task: exec
      command: ${drush.bin}
      arguments:
        - cas-mock-server:start
      options:
        yes: null
        root: ${joinup.site_dir}
    - task: exec
      command: ${drush.bin}
      options:
        yes: null
      arguments:
        - pm:uninstall
        - joinup_eulogin
    - task: exec
      command: ${drush.bin}
      arguments:
        - pm:enable
        - filecache
    - task: run
      command: drupal:settings
      arguments:
        - site-clone
```
The above command also includes some additional commands that are not part of
the deployment process, but are useful for development.

This is the recommended script for local development. You can extend the command
at will but the above is a good mandatory starting point.

After the MySQL container is ready for connections, you can run the following
command to install the site:

```bash
$ docker compose exec --user www-data web ./vendor/bin/run local-deploy
```
Another useful override is also for the clean-install process.

```yaml
  local-install:
    - task: run
      command: toolkit:install-clean
    - task: run
      command: dev:demo-users
    - task: run
      command: dev:install-modules
```

The project offers a `runner.yml.dist` file that can be used as a template and
contains these two commands as a starting point. Copy the file as `runner.yml`
and you will be able to use the commands.

After having the `runner.yml` you will only need the commands `local-install`
and `local-install` to install the site though it is recommended to know what
they really do.

## Accessing the containers

All containers are accessible through the command

```bash
$ docker compose exec my_container_name "/bin/bash"
```

Depending on the container (and its base image), it might be that `/bin/bash` is
not available. In that case, `/bin/sh` and `sh` are good substitutes.

Depending on the user you want to use, you might need to add `--user {username}`
option before the container name, for instance to access the `web` container as
user `www-data`, add `--user www-data`.

## Setup your test files

Run the following command to setup the test files:

```bash
$ docker compose exec --user www-data web ./vendor/bin/run toolkit:build-dev
```
Your `behat.yml` file should now be in the root folder as well as your
`phpcs.xml`.

## Running the tests

### Behat tests

In order to run a behat test, you have to distinguish between the two types of
behat tests we have. The ones running in the clean environment and the ones
running in the cloned environment.
Almost all tests are meant to run in a clean environment. You can run them in
the cloned environment, but results are not guaranteed and also basing your
results on a cloned environment is not guaranting that the tests will pass in
the clean environment.

#### Running the tests in the clean environment

The clean environment is the one that is installed by the
`toolkit:install-clean` or the `local-install` command. In order to run the
tests in this environment, you have to run the following command:

```bash
$ docker compose exec --user www-data web ./vendor/bin/behat --profile=clean --config=behat.yml /tests/features/path/to/feature.feature
```

#### Running the tests in the cloned environment

The difference in the command is that you should not include the
`--profile=clean` option. The command should look like this:

```bash
$ docker compose exec --user www-data web ./vendor/bin/behat --config=behat.yml /tests/features/path/to/feature.feature
```

### PHPUnit tests

The PHPUnit tests are run in the same environment as the Behat tests. The
command is the following:

```bash
$ docker compose exec --user www-data web ./vendor/bin/phpunit path/to/your/test/Class.php
```
The `phpunit.xml` is consumed automatically.

### Step debugging

To use XDEBUG for step debugging, first, make sure that XDEBUG is enabled in
the container and start listening for connections in your IDE. Set a breakpoint
to a related line and run the following command:

```bash
docker compose exec --env XDEBUG_SESSION=1 --user www-data web ./vendor/bin/behat --profile=clean --config=behat.yml ./tests/features/homepage.feature
```
The `XDEBUG_SESSION` environment variable is used to trigger the XDEBUG
session.

PHPUnit can be debugged in the same way.

### Code sniffing

The code sniffing is done with the `phpcs` tool. The configuration file is
`phpcs.xml`. The command to run the code sniffing is the following:

```bash
$ docker compose exec --user www-data web ./vendor/bin/phpcs --parallel=50
```
The default `phpcs.xml` will be generated by the `toolkit:build-dev` command
described in a previous section.

#### Code sniffing pre-push hook

The project contains a pre-push hook that will run the code sniffing before
pushing the code to the remote repository. The hook is installed by the
`toolkit:build-dev` command. The hook will run the code sniffing only on the
files that are staged for commit. If the code sniffing fails, the push will be
aborted.

## Next steps

Please, check the [PhpStorm configuration](./phpstorm_setup.md) for
more information on how to setup your IDE to work with the project. Currently,
the setup is only available for PhpStorm.
