# Collapsible answer block, styled like the theorem/proof boxes in _theorems.scss.
# Usage:
#   {% answer %}          -> summary reads "Answer"
#   {% answer Solution %} -> summary reads "Solution"
# Body is run through the markdown converter, so lists/math/emphasis all work.

module Jekyll
  module Tags
    class AnswerTag < Liquid::Block
      def initialize(tag_name, markup, tokens)
        super
        @caption = markup.strip
      end

      def render(context)
        site = context.registers[:site]
        converter = site.find_converter_instance(::Jekyll::Converters::Markdown)
        caption =
          if @caption.empty?
            "Answer"
          else
            converter.convert(@caption).gsub(/<\/?p[^>]*>/, '').chomp
          end
        body = converter.convert(super(context))
        "<details class=\"answer\">" \
          "<summary>#{caption}</summary>" \
          "<div class=\"theorem-contents\">#{body}</div>" \
        "</details>"
      end
    end
  end
end

Liquid::Template.register_tag('answer', Jekyll::Tags::AnswerTag)
