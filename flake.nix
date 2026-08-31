{
  description = "CLI for Caelestia dots";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";

    caelestia-shell = {
      url = "github:caelestia-dots/shell";
      inputs.nixpkgs.follows = "nixpkgs";
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
      xiu-cli = pkgs.callPackage ./default.nix {
        rev = self.rev or self.dirtyRev;
        caelestia-shell = inputs.caelestia-shell.packages.${pkgs.system}.default;
      };
      with-shell = xiu-cli.override {withShell = true;};
      default = xiu-cli;
    });

    devShells = forAllSystems (pkgs: {
      default = pkgs.mkShellNoCC {
        packages = [self.packages.${pkgs.system}.with-shell];
      };
    });
  };
}
