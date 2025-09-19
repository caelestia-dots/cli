{
  description = "CLI for Caelestia dots";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";

    caelestia-shell = {
      url = "github:caelestia-dots/shell";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.caelestia-cli.follows = "";
    };

    #caelestia-shell = {
    #  url = "github:Av3lle/shell";
    #  inputs.nixpkgs.follows = "nixpkgs";
    #  inputs.caelestia-cli.follows = "";
    #};
  };

  outputs = {
    self,
    nixpkgs,
    nixpkgs-unstable,
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
        caelestia-shell = inputs.caelestia-shell.packages.${pkgs.system}.default;
        app2unit = pkgs.callPackage "${inputs.caelestia-shell}/nix/app2unit.nix" {
          pkgs = inputs.nixpkgs-unstable.legacyPackages.${pkgs.system};
        };
      };
      with-shell = caelestia-cli.override {withShell = true;};
      default = caelestia-cli;
    });

    devShells = forAllSystems (pkgs: {
      default = pkgs.mkShellNoCC {
        packages = [self.packages.${pkgs.system}.with-shell];
      };
    });
  };
}
