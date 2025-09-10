{
  description = "Flake based dev shell for Machine Learning";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = {
    nixpkgs,
    ...
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in {
    devShells.${system}.default =
      pkgs.mkShell
      {
        packages = [
            pkgs.kaggle
          (pkgs.python3.withPackages (p:
            with p; [
                numpy
                requests
                pandas
                numpy
                seaborn
                matplotlib
                jupyterlab
                ipython
                kaggle
            ]))
        ];

        env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.libz
        ];

        shellHook = ''
            echo "Entering ML dev enviroment..." && jupyter lab
        '';
      };
  };
}

