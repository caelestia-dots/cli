return {
    background = "{{ $surface }}",
    foreground = "{{ $onSurface }}",
    cursor     = "{{ $primary }}",
    selection  = "{{ $surfaceContainer }}",
    line_nr    = "{{ $outline }}",
    comment    = "{{ $outlineVariant }}",  -- Make sure this exists!
    
    -- Syntax colors
    keyword    = "{{ $primary }}",
    func       = "{{ $blue }}",
    string     = "{{ $green }}",
    type       = "{{ $mauve }}",
    const      = "{{ $peach }}",
    error      = "{{ $red }}",
    warn       = "{{ $yellow }}"
}
