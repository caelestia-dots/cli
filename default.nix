{
  rev,
  lib,
  python3,
  installShellFiles,
  swappy,
  libnotify,
  slurp,
  wl-clipboard,
  cliphist,
  app2unit,
  dart-sass,
  grim,
  fuzzel,
  gpu-screen-recorder,
  dconf,
  killall,
  caelestia-shell,
  withShell ? false,
  discordBin ? "discord",
  qtctStyle ? "Darkly",
}:
python3.pkgs.buildPythonApplication {
  pname = "caelestia-cli";
  version = "${rev}";
  src = ./.;
  pyproject = true;

  build-system = with python3.pkgs; [
    hatch-vcs
    hatchling
  ];

  dependencies = with python3.pkgs; [
    materialyoucolor
    pillow
  ];

  pythonImportsCheck = ["caelestia"];

  nativeBuildInputs = [installShellFiles];
  propagatedBuildInputs =
    [
      swappy
      libnotify
      slurp
      wl-clipboard
      cliphist
      app2unit
      dart-sass
      grim
      fuzzel
      gpu-screen-recorder
      dconf
      killall
    ]
    ++ lib.optional withShell caelestia-shell;

  SETUPTOOLS_SCM_PRETEND_VERSION = 1;

  patchPhase = ''
    # Replace qs config call with nix shell pkg bin
    substituteInPlace src/caelestia/subcommands/shell.py \
    	--replace-fail '"qs", "-c", "caelestia"' '"caelestia-shell"'
    substituteInPlace src/caelestia/subcommands/screenshot.py \
    	--replace-fail '"qs", "-c", "caelestia"' '"caelestia-shell"'

    # Use config bin instead of discord + fix todoist + fix app2unit
    substituteInPlace src/caelestia/subcommands/toggle.py \
    	--replace-fail 'discord' ${discordBin} \
      --replace-fail 'todoist' 'todoist.desktop'\
      --replace-fail 'app2unit' ${app2unit}/bin/app2unit

    # Use config style instead of darkly
    substituteInPlace src/caelestia/data/templates/qtct.conf \
    	--replace-fail 'Darkly' '${qtctStyle}'
  '';

  postInstall = "installShellCompletion completions/caelestia.fish";

  meta = {
    description = "The main control script for the Caelestia dotfiles";
    homepage = "https://github.com/caelestia-dots/cli";
    license = lib.licenses.gpl3Only;
    mainProgram = "caelestia";
    platforms = lib.platforms.linux;
  };
}
