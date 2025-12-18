let
  inputs = import ../../../npins;
  # This specific version of nixpkgs is needed to get mysql 8.4.3 with which we 
  # performed all of our experiments.
  pkgs = import inputs.nixpkgs_mysql_8_4_3 { };
  lib = pkgs.lib;

  fs = lib.fileset;
  sourceFiles = ./skeletons;
  src_grammars = fs.toSource {
    root = ./.;
    fileset = sourceFiles;
  };

  custom-bison = pkgs.bison.overrideAttrs (
    p: final: {
      postInstall = ''
        cp -vrT ${./skeletons} $out/share/bison/skeletons/
      '';
      doInstallCheck = false; # Disabled tests, they are too long when debuging
    }
  );
in
(pkgs.mysql84.overrideAttrs (
  final: prev: {
    patches = prev.patches ++ [ ./sql_yacc.patch ];
  }
)).override
  ({ bison = custom-bison; })