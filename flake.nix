{
  description = "Caelestia CLI";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs-stable.url = "github:NixOS/nixpkgs/23.11";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { ... } @ inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
    let
      stable-packages = final: _prev: {
        stable = import inputs.nixpkgs-stable {
          system = final.system;
          config.allowUnfree = true;
        };
      };

      pkgs = import inputs.nixpkgs {
        inherit system;
        overlays = [
          stable-packages
        ];
      };

      caelestia = pkgs.python313Packages.buildPythonPackage rec {
        name = "caelestia-cli";
        version = "v0.0.1";
        src = ./.;
        pyproject = true;

        SETUPTOOLS_SCM_PRETEND_VERSION = version;

        propagatedBuildInputs = with pkgs.python313Packages; [
          hatch-vcs
          hatchling
          materialyoucolor
          pillow
          pkgs.cliphist
          pkgs.dbus
          pkgs.fuzzel
          pkgs.glib
          pkgs.grim
          pkgs.hatch
          pkgs.libnotify
          pkgs.procps
          pkgs.pulseaudio
          pkgs.slurp
          pkgs.swappy
          pkgs.wl-clipboard-rs
          pkgs.wl-screenrec
        ];
      };
    in
    {
      packages = {
        default = caelestia;
      };

      devShells.default = pkgs.mkShell {
        packages =
          with pkgs; [ ]
          ++ pkgs.lib.optionals pkgs.stdenv.isDarwin (with pkgs; [
          ]);
      };
    });
}

