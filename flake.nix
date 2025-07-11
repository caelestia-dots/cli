{
  description = "Caelstia Cli Packaged as flake";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        deps = with pkgs; [
          libnotify
          swappy
          grim
          dart-sass
          app2unit
          wl-clipboard
          slurp
          wl-screenrec
          libpulseaudio
          cliphist
          fuzzel
          killall
          python3Packages.hatch-vcs
          python3Packages.pillow
          python3Packages.materialyoucolor
        ];
      in
      {
        packages.default = pkgs.python3Packages.buildPythonPackage {
          pname = "caelestia-cli";
          src = ./.;
          version = "0.0.1+git.${self.shortRev or "dirty"}";
          pyproject = true;

          build-system = with pkgs.python3Packages; [
            hatchling
          ];

          dependencies = deps;
        };
      }
    );
}
