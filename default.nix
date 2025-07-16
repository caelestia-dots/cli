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
  wl-screenrec,
  dconf,
  killall,
  caelestia-shell,
  withShell ? false,
  discordBin ? "discord",
  qtctStyle ? "Fusion",
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
      wl-screenrec
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

    # Use config bin instead of discord + fix todoist
    substituteInPlace src/caelestia/subcommands/toggle.py \
    	--replace-fail 'discord' ${discordBin} \
      --replace-fail 'todoist' 'todoist.desktop'

    # Use config style instead of fusion
    substituteInPlace src/caelestia/data/templates/qtct.conf \
    	--replace-fail 'Fusion' '${qtctStyle}'
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
