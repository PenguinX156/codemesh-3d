/** Required notice: Meta's Contributor tier may train on inputs. Shown wherever files can be uploaded. */
export function Disclosure() {
  return (
    <aside className="callout" aria-label="Data disclosure">
      <strong>Before you upload.</strong> Processing uses Meta&apos;s Muse Spark Contributor tier and SAM 3.1.
      Contributor inputs may be used to train Meta models &mdash; don&apos;t upload personal or sensitive content
      (people, faces, documents, private spaces).
    </aside>
  )
}
