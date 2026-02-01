import json
import random
from argparse import Namespace

from caelestia.utils.scheme import (
    Scheme,
    get_scheme,
    get_scheme_flavours,
    get_scheme_modes,
    get_scheme_names,
    scheme_variants,
)
from caelestia.utils.theme import apply_colours


class Set:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def _apply_and_save(self, scheme: Scheme, snapshot: dict) -> None:
        try:
            apply_colours(scheme.colours, scheme.mode)
            scheme.save()
        except Exception:
            scheme._name = snapshot["name"]
            scheme._flavour = snapshot["flavour"]
            scheme._mode = snapshot["mode"]
            scheme._variant = snapshot["variant"]
            if "colours" in snapshot:
                scheme._colours = snapshot["colours"]
            raise

    def run(self) -> None:
        scheme = get_scheme()

        if self.args.notify:
            scheme.notify = True

        if self.args.random:
            snapshot = {
                "name": scheme._name,
                "flavour": scheme._flavour,
                "mode": scheme._mode,
                "variant": scheme._variant,
                "colours": scheme._colours,
            }
            
            scheme._name = random.choice(get_scheme_names())
            scheme._flavour = random.choice(get_scheme_flavours(scheme._name))
            scheme._mode = random.choice(get_scheme_modes(scheme._name, scheme._flavour))
            scheme._update_colours()
            
            self._apply_and_save(scheme, snapshot)
        elif self.args.name or self.args.flavour or self.args.mode or self.args.variant:
            snapshot = {
                "name": scheme._name,
                "flavour": scheme._flavour,
                "mode": scheme._mode,
                "variant": scheme._variant,
            }
            
            if self.args.name:
                scheme._name = self.args.name
                scheme._check_flavour()
                scheme._check_mode()
            if self.args.flavour:
                scheme._flavour = self.args.flavour
                scheme._check_mode()
            if self.args.mode:
                scheme._mode = self.args.mode
            if self.args.variant:
                scheme._variant = self.args.variant
            
            scheme._update_colours()
            self._apply_and_save(scheme, snapshot)
        else:
            print("No args given. Use --name, --flavour, --mode, --variant or --random to set a scheme")


class Get:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        scheme = get_scheme()

        if self.args.name or self.args.flavour or self.args.mode or self.args.variant:
            if self.args.name:
                print(scheme.name)
            if self.args.flavour:
                print(scheme.flavour)
            if self.args.mode:
                print(scheme.mode)
            if self.args.variant:
                print(scheme.variant)
        else:
            print(scheme)


class List:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        multiple = [self.args.names, self.args.flavours, self.args.modes, self.args.variants].count(True) > 1

        if self.args.names or self.args.flavours or self.args.modes or self.args.variants:
            if self.args.names:
                if multiple:
                    print("Names:", *get_scheme_names())
                else:
                    print("\n".join(get_scheme_names()))
            if self.args.flavours:
                if multiple:
                    print("Flavours:", *get_scheme_flavours())
                else:
                    print("\n".join(get_scheme_flavours()))
            if self.args.modes:
                if multiple:
                    print("Modes:", *get_scheme_modes())
                else:
                    print("\n".join(get_scheme_modes()))
            if self.args.variants:
                if multiple:
                    print("Variants:", *scheme_variants)
                else:
                    print("\n".join(scheme_variants))
        else:
            current_scheme = get_scheme()
            schemes = {}
            for scheme in get_scheme_names():
                schemes[scheme] = {}
                for flavour in get_scheme_flavours(scheme):
                    s = Scheme(
                        {
                            "name": scheme,
                            "flavour": flavour,
                            "mode": current_scheme.mode,
                            "variant": current_scheme.variant,
                            "colours": current_scheme.colours,
                        }
                    )
                    modes = get_scheme_modes(scheme, flavour)
                    if s.mode not in modes:
                        s._mode = modes[0]
                    try:
                        s._update_colours()
                        schemes[scheme][flavour] = s.colours
                    except ValueError:
                        pass

            print(json.dumps(schemes))
