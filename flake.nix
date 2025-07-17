{
  description = "CLI for Caelestia dots";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";

    app2unit = {
      url = "github:soramanew/app2unit";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    caelestia-shell = {
      url = "github:caelestia-dots/shell";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.app2unit.follows = "app2unit";
      inputs.caelestia-cli.follows = "";
    };
  };

  outputs = {
    self,
    nixpkgs,
    ...
  } @ inputs: let
    forAllSystems = fn:
      nixpkgs.lib.genAttrs nixpkgs.lib.platforms.linux (
        system: fn nixpkgs.legacyPackages.${system}
      );
  in {
    formatter = forAllSystems (pkgs: pkgs.alejandra);

    packages = forAllSystems (pkgs: rec {
      caelestia-cli = pkgs.callPackage ./default.nix {
        rev = self.rev or self.dirtyRev;
        app2unit = inputs.app2unit.packages.${pkgs.system}.default;
        caelestia-shell = inputs.caelestia-shell.packages.${pkgs.system}.default;
      };
      default = caelestia-cli;
    });

    devShells = forAllSystems (pkgs: {
      default = pkgs.mkShellNoCC {
        inputsFrom = [self.packages.${pkgs.system}.caelestia-cli];
        packages = [
          (pkgs.writeShellScriptBin "caelestia" ''
            cd src && python -m caelestia "$@"
          '')
        ];
      };
    });
  };
}
