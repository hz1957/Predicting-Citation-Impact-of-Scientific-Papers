#!/bin/bash
# Rebuild preview.html from midterm_unsupervised_section.html plus the
# local-only k-explorer fragment (k_explorer_fragment.html).
cd "$(dirname "$0")"
{
  sed -n '1,/<main>/p' preview.html
  sed 's|\.\./\.\./assets/img/midterm/|../outputs/web/|g' midterm_unsupervised_section.html
  if [ -f k_explorer_meta.json ]; then
    printf '<script>window.KX_META = %s;</script>\n' "$(cat k_explorer_meta.json)"
  fi
  cat k_explorer_fragment.html
  printf '  </main>\n</body>\n</html>\n'
} > preview.html.tmp && mv preview.html.tmp preview.html
echo "preview.html rebuilt"
