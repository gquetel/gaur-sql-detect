let
  inputs = import ./npins;
  pkgs = import inputs.nixpkgs {
    config.allowUnfree = true;
  };

  mysql-connector =
    let
      pname = "mysql-connector-python";
      version = "9.3.0";
      format = "wheel";
    in
    pkgs.python313.pkgs.buildPythonPackage {
      # Have to use direct fetchurl as package is not updated in nixkpgs
      inherit pname version format;
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/23/1d/8c2c6672094b538f4881f7714e5332fdcddd05a7e196cbc9eb4a9b5e9a45/mysql_connector_python-9.3.0-py2.py3-none-any.whl";
        sha256 = "sha256-irdxnWFM9UY1IQgvq4avwhraUEtTgWYJDgDuqh/3Kbw=";
      };
      doCheck = false;
    };
  pythonEnv = (
    (pkgs.python313.withPackages (
      ps:
      [
        ps.pandas
        ps.numpy
        ps.tqdm
        ps.zstandard # for gaur cache feature compression

        # Notebooks & Visualisation
        ps.ipykernel
        ps.jupyter
        ps.kaleido
        ps.matplotlib
        ps.plotly
        ps.tabulate

        # Models
        ps.accelerate
        ps.evaluate
        ps.scikit-learn
        ps.scipy
        ps.sentence-transformers
        ps.torch
        ps.transformers
      ]
      ++ [ mysql-connector ]
    )).override
      (args: {
        ignoreCollisions = true;
      })
  );
in
pkgs.mkShell rec {
  packages = [
    pythonEnv
  ];

  allowUnfree = true;
  catchConflicts = false;
  shellHook = ''
    export CUSTOM_INTERPRETER_PATH="${pythonEnv}/bin/python"
  '';
}
