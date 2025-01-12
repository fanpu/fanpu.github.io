---
title: "Creating Trackback Requests for Static Sites"
layout: post
tags: [code, general]
cover: fuji_ascent_1.webp
cover_preview: fuji_ascent_1.webp
caption: Yoshida Trail ascent on Mt Fuji, Shizuoka Prefecture, Japan
class: post-template
author: fanpu
toc:
  sidebar: left
giscus_comments: true
description: >
  A simple guide on creating manual Trackback requests for static sites
  to increase visibility and discoverability
published: true
---

In this article, I will show you how you can generate trackback requests to
external websites to link back to your static site like
[Jekyll](https://jekyllrb.com/) or [Hugo](https://gohugo.io/).
I decided to write this article after realizing how there is almost no
information online about how to make DIY trackback requests when I was
trying to set it up.

## What is Trackback?
From [Wikipedia]():

> A trackback allows one website to notify another about an update. It is one
of four types of linkback methods for website authors to request notification
when somebody links to one of their documents. This enables authors to keep
track of who is linking to their articles. Some weblog software, such as
SilverStripe, WordPress, Drupal, and Movable Type, supports automatic pingbacks
where all the links in a published article can be pinged when the article is
published. The term is used colloquially for any kind of linkback.

Essentially, it is a mechanism for other websites to know that you mentioned them,
with the hope that they'll notice you and possibly mention you as well.
It helps to increase the visibility and discoverability of your website.

## Use Case
My use case was to send trackbacks to [arXiv](https://info.arxiv.org/help/trackback.html),
so that specific arXiv papers will know that my blog post mentioned them, and readers
can also check it out as an additional resource. In particular, each of my [paper summary]({% link _pages/summaries.md %}) posts is based around a paper, and it would be nice if they could be linked from the respective arXiv paper abstract pages.

In arXiv, there is a blog link section that will track websites that made trackback
requests for a given paper:

{% include figure.html 
    path="/assets/img/posts/trackback_post/trackback_blog_link.webp"
    width="600px"
    class="z-depth-1"
%}

Unfortunately, if you try to search for anything about trackbacks and/or
pingbacks, most of what you'll get are articles about how to disable them on
popular blogging platforms like WordPress due to [widespread misuse and spam](https://blog.hubspot.com/website/trackback-spam), or
otherwise how to configure them. 

There was also a 7-year old [StackOverflow post](https://superuser.com/questions/1098682/how-to-send-trackback-to-arxiv-papers-from-a-jekyll-blog) about how to
create trackback requests for arXiv, essentially the same problem I was facing. 
Sadly, it currently has a grand total of 0 answers and 0 comments. I hope this
article might be useful if the author is still facing the issue.

## Manually Creating Trackback Requests
The convenience of CMS blogging software like
[WordPress](https://wordpress.org/documentation/article/trackbacks-and-pingbacks/)
is that it supports features like automated trackbacks and pingbacks for content
that you create. Static site generators are not capable of this, since by design they
are static and stateless. This means that we have to make such requests manually,
which is fortunately not too difficult!

Here's a very simple script for doing it. In this example, the target URL
is for the arXiv trackback endpoint.

Before reading or running the code, please note that you **SHOULD NOT** test or
experiment on this with trackback listener URLs and spam them. You should only
make requests if they are legitimate and you have a genuine reason for letting
them know about your blog post.  Trackback spam is a serious issue and part of
why they have become so unpopular and unmanageable is due to the high volumes of
spam.

{% highlight python linenos %}
{% raw %}
import requests

# Replace with your own data
data = {
    'title': 'My Awesome Blog Post',
    'url': 'https://my-blog.com/post/',
    'blog_name': 'My Awesome Blog'
}

# Replace with actual Trackback destination URL
trackback_url = f'https://foo.bar/trackback/post_id'

response = requests.post(trackback_url, data=data)

if response.status_code == 200:
    print("Trackback successful!")
else:
    print(f"Trackback failed with status code: {response.status_code}")

print(response.content.decode())
{% endraw %}
{% endhighlight %}

A successful response has the `error` field set to `0`:

{% highlight xml %}
{% raw %}
<?xml version="1.0" encoding="utf-8"?>
<response>
  <error>0</error>
</response>
{% endraw %}
{% endhighlight %}

If an error occured, the `error` field is set to `1`:

{% highlight xml %}
{% raw %}
<?xml version="1.0" encoding="utf-8"?>
<response>
  <error>1</error>
  <message>(some error message)</message>
</response>
{% endraw %}
{% endhighlight %}

## Conclusion
And that's all there is to creating Trackback requests! It's actually quite
simple, and is just not terribly well-documented. 

As a final parting word, a reminder again to please use it *responsibly* and *stay
away from any behavior that could be constituted as spamming*.