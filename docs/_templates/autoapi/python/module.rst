{# Pelagos-Py override of the stock sphinx-autoapi module template.
   Differences from the default:
     * the module docstring is NOT rendered (kept out of the page),
     * no Attributes/Classes/Functions summary tables,
     * no "Module Contents" heading.
   The page is just the module title followed by its documented members. #}
{% if obj.display %}
   {% if is_own_page %}
{{ obj.id }}
{{ "=" * obj.id|length }}

.. py:module:: {{ obj.name }}

      {% block submodules %}
         {% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
         {% set visible_submodules = obj.submodules|selectattr("display")|list %}
         {% set visible_submodules = (visible_subpackages + visible_submodules)|sort %}
         {% if visible_submodules %}
Submodules
----------

.. toctree::
   :maxdepth: 1

            {% for submodule in visible_submodules %}
   {{ submodule.include_path }}
            {% endfor %}


         {% endif %}
      {% endblock %}
      {% block content %}
         {% set visible_children = obj.children|selectattr("display")|list %}
         {% for obj_item in visible_children %}
{{ obj_item.render()|indent(0) }}
         {% endfor %}
      {% endblock %}
   {% else %}
.. py:module:: {{ obj.name }}

      {% set visible_children = obj.children|selectattr("display")|list %}
      {% for obj_item in visible_children %}
   {{ obj_item.render()|indent(3) }}
      {% endfor %}
   {% endif %}
{% endif %}
