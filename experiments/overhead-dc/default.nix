let
  inputs = import ../../npins;
  pkgs = import inputs.nixpkgs_mysql_8_4_3 { };

  customMysql = import ./instrumented-mysql;
  init_db_sql = import ./init_db_script;
  standardMysql = pkgs.mysql84;

  # fetchzip rely on extension, file is provided without one, we add a dummy GET parameter
  # to allow the fetcher to detect this is a zip file. See:
  # https://discourse.nixos.org/t/fetchzip-fire-without-extension/36569/3
  anubis = pkgs.fetchzip {
    # url = "https://zenodo.org/api/records/17086037/files-archive?=.zip";
    # I love zenodo but download is just so slow.
    url = "https://www.kaggle.com/api/v1/datasets/download/grgorqutel/superviz25-sql-injection-detection-dataset#.zip";
    hash = "sha256-NZnK7XByaymM/uVJ2mmZadgf2jkgcRPVPlUBH/fNy5I=";
  };

  mysql-connector =
    let
      pname = "mysql-connector-python";
      version = "9.3.0";
      format = "wheel";
    in
    pkgs.python312.pkgs.buildPythonPackage {
      # Have to use direct fetchurl as package is not updated in nixkpgs
      inherit pname version format;
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/23/1d/8c2c6672094b538f4881f7714e5332fdcddd05a7e196cbc9eb4a9b5e9a45/mysql_connector_python-9.3.0-py2.py3-none-any.whl";
        sha256 = "sha256-irdxnWFM9UY1IQgvq4avwhraUEtTgWYJDgDuqh/3Kbw=";
      };
      doCheck = false;
    };

  pythonEnv = pkgs.buildEnv {
    name = "pyenv";
    paths = [
      (pkgs.python312.withPackages (ps: [
        ps.pandas
        ps.numpy
        ps.tqdm
        ps.plotly
        ps.scipy
        mysql-connector
      ]))
      pkgs.percona-toolkit
    ];
  };
in

pkgs.writeScriptBin "overhead-experiment" ''
  #!/usr/bin/env bash
  # -------------------------- Instrumented SQL --------------------------

  # Create tmpdir for experiments and automatically exist it. 
  CUSTOM_TMPDIR="$HOME/tmp/overhead-custom/"
  mkdir -p "$CUSTOM_TMPDIR" 
  rm -rf "$CUSTOM_TMPDIR/*"

  CUSTOM_DATADIR="$CUSTOM_TMPDIR/custom_mysql"
  CUSTOM_SOCKET="$CUSTOM_TMPDIR/socket"
  CUSTOM_PORT=61690
  CUSTOM_CSV="./overhead_custom.csv"

  # Initialize Instrumented MySQL server
  ${customMysql}/bin/mysqld --initialize-insecure \
    --datadir="$CUSTOM_DATADIR" --basedir="${customMysql}/bin/"

  # Start Instrumented MySQL server
  # Disabling secure-file-priv allows to load csv files using load data stmt. 

  ${customMysql}/bin/mysqld --log-error --basedir="${customMysql}/bin/" --socket "$CUSTOM_SOCKET" --datadir "$CUSTOM_DATADIR" --port "$CUSTOM_PORT" --daemonize --secure-file-priv=""
  CUSTOM_PID=$!

  echo "Waiting for server to start"
  sleep 2

  # Set password, and initialize db with test_db
  ${customMysql}/bin/mysql --user=root --socket "$CUSTOM_SOCKET" < ${init_db_sql}

  # -------------------------- Normal SQL --------------------------
  # Create tmpdir for experiments and automatically exist it. 
  NORMAL_TMPDIR="$HOME/tmp/overhead-normal/"
  mkdir -p "$NORMAL_TMPDIR" 

  # When crash, folder still exists, we make sure it's empty.
  rm -rf "$NORMAL_TMPDIR/*"

  NORMAL_DATADIR="$NORMAL_TMPDIR/normal_mysql"
  NORMAL_SOCKET="$NORMAL_TMPDIR/socket"
  NORMAL_PORT=61691
  NORMAL_CSV="./overhead_normal.csv"

  # Initialize Instrumented MySQL server
  ${pkgs.mysql84}/bin/mysqld --initialize-insecure \
    --datadir="$NORMAL_DATADIR" --basedir="${pkgs.mysql84}/bin/"

  # Start Instrumented MySQL server

  ${pkgs.mysql84}/bin/mysqld --log-error --basedir="${pkgs.mysql84}/bin/" --socket "$NORMAL_SOCKET" --datadir "$NORMAL_DATADIR" --port "$NORMAL_PORT" --daemonize --secure-file-priv=""
  NORMAL_PID=$!
  echo "Waiting for server to start"
  sleep 2

  # Set password, and initialize db with test_db
  ${pkgs.mysql84}/bin/mysql --user=root --socket "$NORMAL_SOCKET" < ${init_db_sql}

  # -------------------------- Experiments --------------------------
  # Calling the script such a way, allows to have access to pt-kill command in script.
  export PATH="${pythonEnv}/bin:$PATH" 
  python3 ./overhead.py --nsckt "$NORMAL_SOCKET" --csckt "$CUSTOM_SOCKET" --dataset "${anubis}/dataset.csv" --testing

  # Attempt to shut down, else force shutdown. 
  ${pkgs.mysql84}/bin/mysqladmin --socket="$NORMAL_SOCKET" -u root --password="root" shutdown || kill $NORMAL_PID
  ${customMysql}/bin/mysqladmin --socket="$CUSTOM_SOCKET" -u root --password="root" shutdown || kill $CUSTOM_PID

  rm -rf $NORMAL_TMPDIR $CUSTOM_TMPDIR
''